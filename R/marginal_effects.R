#' Compute marginal effects from a fitted model
#'
#' Generic function; see [marginal_effects.jocf()] for the jocf method.
#'
#' @param object A fitted model object.
#' @param ... Additional arguments passed to the method.
#'
#' @return Method-specific; see the relevant method documentation.
#'
#' @examples
#' ## See marginal_effects.jocf() for a full example
#' set.seed(42)
#' n <- 150
#' X <- matrix(rnorm(n * 3), ncol = 3)
#' Y <- sample(1:3, n, replace = TRUE, prob = c(0.3, 0.5, 0.2))
#' fit <- jocf(Y, X, num.trees = 50)
#' me <- marginal_effects(fit, X)
#' me
#'
#' @export
marginal_effects <- function(object, ...) UseMethod("marginal_effects")


#' Nonparametric marginal effects for a fitted jocf object
#'
#' Computes average marginal effects (AME) following Lechner (2019).
#' For each target covariate \eqn{j} and each evaluation point \eqn{w_i}:
#'
#' * **Continuous** covariate \eqn{j}:
#'   \deqn{\widehat{\partial}_j p_m(w_i) =
#'     \frac{\hat{p}_m(w_i + \omega\,\hat{\sigma}_j\,e_j)
#'           - \hat{p}_m(w_i - \omega\,\hat{\sigma}_j\,e_j)}
#'          {2\,\omega\,\hat{\sigma}_j}}
#'   where \eqn{\hat{\sigma}_j} is the sample standard deviation of covariate
#'   \eqn{j} and \eqn{e_j} is the \eqn{j}-th unit vector.
#'
#' * **Discrete** covariate \eqn{j}:
#'   \deqn{\widehat{\partial}_j p_m(w_i) =
#'     \hat{p}_m(w_i \text{ with } x_j = \lfloor x_{ij} \rfloor + 1)
#'     - \hat{p}_m(w_i \text{ with } x_j = \lfloor x_{ij} \rfloor)}
#'
#' The AME for covariate \eqn{j} and class \eqn{m} is the average of these
#' individual effects over all evaluation points.
#'
#' @param object A fitted object of class `"jocf"`.
#' @param data Numeric matrix or data.frame with `object$k` columns.  Typically
#'   the training data, but any evaluation data with the same column layout is
#'   accepted.
#' @param eval Character; where to evaluate the marginal effects.
#'   `"mean"` (default) averages individual effects over all rows of `data`;
#'   `"atmean"` evaluates at the column means; `"atmedian"` evaluates at the
#'   column medians.
#' @param target_covariates Integer vector of 1-based column indices selecting
#'   which covariates to differentiate.  `NULL` (default) uses all columns.
#' @param discrete_vars Integer vector of 1-based column indices identifying
#'   discrete covariates.  Default `NULL` (all continuous).
#' @param omega Positive scalar bandwidth for the continuous finite difference.
#'   Default `0.001`.
#' @param num.threads Positive integer or `NULL`. Number of OpenMP threads.
#'   `NULL` (default) uses all available cores.
#' @param ... Currently unused.
#'
#' @return An object of class `"jocf_me"` — a list with components:
#' \describe{
#'   \item{`AME`}{(k_target x M) numeric matrix.  Row \eqn{j}, column \eqn{m}
#'     is the AME of that covariate on \eqn{P(Y = m)}.
#'     Rows sum to zero across classes by construction.}
#'   \item{`eval`}{The `eval` argument used.}
#'   \item{`target_covariates`}{1-based integer vector of differentiated covariates.}
#'   \item{`discrete_vars`}{Integer vector passed as `discrete_vars`.}
#'   \item{`omega`}{Bandwidth used for continuous variables.}
#'   \item{`call`}{The matched call.}
#' }
#'
#' @examples
#' ## Simulate data and fit a forest
#' set.seed(42)
#' n <- 150
#' X <- matrix(rnorm(n * 3), ncol = 3)
#' Y <- sample(1:3, n, replace = TRUE, prob = c(0.3, 0.5, 0.2))
#' fit <- jocf(Y, X, num.trees = 50)
#'
#' ## Average marginal effects (all covariates)
#' me <- marginal_effects(fit, X)
#' me
#'
#' ## Evaluate at the mean and for selected covariates only
#' me_atmean <- marginal_effects(fit, X, eval = "atmean",
#'                               target_covariates = c(1, 3))
#' me_atmean
#'
#' @export
marginal_effects.jocf <- function(object,
                                   data,
                                   eval              = c("mean", "atmean", "atmedian"),
                                   target_covariates = NULL,
                                   discrete_vars     = NULL,
                                   omega             = 0.001,
                                   num.threads       = NULL,
                                   ...) {
  cl   <- match.call()
  eval <- match.arg(eval)

  if (!inherits(object, "jocf"))
    stop('`object` must be of class "jocf".', call. = FALSE)

  data <- as.matrix(data)
  if (!is.numeric(data))
    stop("`data` must be numeric.", call. = FALSE)
  if (ncol(data) != object$k)
    stop(sprintf("`data` must have %d column(s) (same as training X).",
                 object$k), call. = FALSE)
  if (anyNA(data))
    stop("Missing values are not allowed in `data`.", call. = FALSE)

  if (!is.numeric(omega) || length(omega) != 1L || omega <= 0)
    stop("`omega` must be a single positive number.", call. = FALSE)

  k <- ncol(data)
  M <- object$M

  # Resolve target_covariates to a sorted, unique, 1-based integer vector
  if (is.null(target_covariates)) {
    target_1based <- seq_len(k)
  } else {
    target_1based <- sort(unique(as.integer(target_covariates)))
    if (any(target_1based < 1L) || any(target_1based > k))
      stop("`target_covariates` values must be in 1..ncol(data).", call. = FALSE)
  }
  target_0based <- target_1based - 1L
  k_target      <- length(target_1based)

  # Discrete indicator aligned with target_covariates
  is_discrete_target <- logical(k_target)
  if (!is.null(discrete_vars)) {
    discrete_vars <- as.integer(discrete_vars)
    if (any(discrete_vars < 1L) || any(discrete_vars > k))
      stop("`discrete_vars` values must be in 1..ncol(data).", call. = FALSE)
    is_discrete_target <- target_1based %in% discrete_vars
  }

  # Step sizes for continuous covariates (omega * sample SD)
  sds         <- apply(data, 2, sd)
  sds[sds == 0] <- 1.0   # guard against zero-variance columns
  h_vec       <- omega * sds[target_1based]

  # Build the evaluation matrix
  X_eval <- switch(eval,
    "mean"     = data,
    "atmean"   = matrix(colMeans(data),          nrow = 1L),
    "atmedian" = matrix(apply(data, 2, median),  nrow = 1L)
  )

  # Delegate to C++ engine
  AME <- marginal_effects_cpp(
    forest      = object$forest,
    X_eval      = X_eval,
    target_vars = target_0based,
    is_discrete = is_discrete_target,
    h_vec       = h_vec,
    M           = M,
    num_threads = resolve_num_threads(num.threads)
  )

  # Attach names
  col_nms <- colnames(data)
  if (is.null(col_nms)) col_nms <- paste0("X", seq_len(k))
  rownames(AME) <- col_nms[target_1based]
  colnames(AME) <- paste0("P(Y=", seq_len(M), ")")

  structure(
    list(
      AME               = AME,
      eval              = eval,
      target_covariates = target_1based,
      discrete_vars     = discrete_vars,
      omega             = omega,
      call              = cl
    ),
    class = "jocf_me"
  )
}


#' Print method for jocf_me objects
#'
#' @param x A `"jocf_me"` object.
#' @param digits Number of significant digits to display. Default `4`.
#' @param ... Currently unused.
#'
#' @return Invisibly returns `x`.
#' @export
print.jocf_me <- function(x, digits = 4L, ...) {
  M <- ncol(x$AME)
  k <- nrow(x$AME)
  cat(sprintf("Average Marginal Effects \u2014 jocf  [eval = \"%s\"]\n", x$eval))
  cat(sprintf("  %d covariate(s), %d outcome class(es)\n\n", k, M))
  mat <- x$AME
  if (is.null(rownames(mat))) rownames(mat) <- paste0("X", seq_len(k))
  print(round(mat, digits))
  invisible(x)
}
