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
#' Computes marginal effects via finite differences.
#' For each target covariate \eqn{j} and each evaluation point \eqn{w_i}:
#'
#' * **Continuous** covariate \eqn{j}:
#'   \deqn{\widehat{\partial}_j p_m(w_i) =
#'     \frac{\hat{p}_m(w_i + h\,\hat{\sigma}_j\,e_j)
#'           - \hat{p}_m(w_i - h\,\hat{\sigma}_j\,e_j)}
#'          {2\,h\,\hat{\sigma}_j}}
#'   where \eqn{\hat{\sigma}_j} is the sample standard deviation of covariate
#'   \eqn{j} and \eqn{e_j} is the \eqn{j}-th unit vector.
#'
#' * **Discrete** covariate \eqn{j}:
#'   \deqn{\widehat{\partial}_j p_m(w_i) =
#'     \hat{p}_m(w_i \text{ with } x_j = \lfloor x_{ij} \rfloor + 1)
#'     - \hat{p}_m(w_i \text{ with } x_j = \lfloor x_{ij} \rfloor)}
#'
#' The marginal effect for covariate \eqn{j} and class \eqn{m} is the average
#' of these individual effects over all evaluation points.
#'
#' Factor and logical covariates are automatically excluded because finite
#' differences are not meaningful for categorical variables.  A `message()` is
#' printed when columns are excluded; explicitly requesting a factor column via
#' `target_covariates` produces a `warning()`.
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
#' @param bandwidth Positive scalar bandwidth for the continuous finite
#'   difference.  Default `0.001`.
#' @param num.threads Positive integer or `NULL`. Number of OpenMP threads.
#'   `NULL` (default) uses all available cores.
#' @param ... Currently unused.
#'
#' @return An object of class `"jocf_me"` — a list with components:
#' \describe{
#'   \item{`effects`}{(k_target x M) numeric matrix.  Row \eqn{j}, column
#'     \eqn{m} is the marginal effect of that covariate on \eqn{P(Y = m)}.
#'     Rows sum to zero across classes by construction.}
#'   \item{`std.error`}{(k_target x M) numeric matrix of standard errors
#'     (only when the model was fit with `honesty = TRUE`; `NULL` otherwise).}
#'   \item{`ci.lower`}{(k_target x M) numeric matrix of lower bounds of 95\%
#'     confidence intervals (`NULL` when non-honest).}
#'   \item{`ci.upper`}{(k_target x M) numeric matrix of upper bounds of 95\%
#'     confidence intervals (`NULL` when non-honest).}
#'   \item{`eval`}{The `eval` argument used.}
#'   \item{`target_covariates`}{1-based integer vector of differentiated covariates.}
#'   \item{`discrete_vars`}{Integer vector passed as `discrete_vars`.}
#'   \item{`bandwidth`}{Bandwidth used for continuous variables.}
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
#' \donttest{
#' ## Honest forest: marginal effects with standard errors
#' fit_h <- jocf(Y, X, num.trees = 50, honesty = TRUE)
#' me_h <- marginal_effects(fit_h, X)
#' me_h           # prints effects + standard errors + 95% CIs
#' me_h$std.error # (k_target x M) matrix of SEs
#' }
#'
#' @export
marginal_effects.jocf <- function(object,
                                   data,
                                   eval              = c("mean", "atmean", "atmedian"),
                                   target_covariates = NULL,
                                   discrete_vars     = NULL,
                                   bandwidth         = 0.001,
                                   num.threads       = NULL,
                                   ...) {
  cl   <- match.call()
  eval <- match.arg(eval)

  if (!inherits(object, "jocf"))
    stop('`object` must be of class "jocf".', call. = FALSE)

  # Encode factors using stored training-time metadata
  encoded <- encode_factors(data, factor_info = object$factor_info)
  data <- encoded$X_encoded
  if (ncol(data) != object$k)
    stop(sprintf("`data` must have %d column(s) (same as training X).",
                 object$k), call. = FALSE)
  if (anyNA(data))
    stop("Missing values are not allowed in `data`.", call. = FALSE)

  if (!is.numeric(bandwidth) || length(bandwidth) != 1L || bandwidth <= 0)
    stop("`bandwidth` must be a single positive number.", call. = FALSE)

  k <- ncol(data)
  M <- object$M

  # Identify factor columns (finite-difference is not meaningful for these)
  factor_cols <- integer(0)
  if (!is.null(object$factor_info)) {
    factor_cols <- which(vapply(object$factor_info, Negate(is.null), logical(1)))
  }

  # Resolve target_covariates to a sorted, unique, 1-based integer vector
  if (is.null(target_covariates)) {
    target_1based <- seq_len(k)
    # Auto-exclude factor columns
    if (length(factor_cols) > 0) {
      target_1based <- setdiff(target_1based, factor_cols)
      col_nms <- colnames(data)
      if (is.null(col_nms)) col_nms <- paste0("X", seq_len(k))
      message(sprintf(
        "Factor/logical covariate(s) %s excluded from marginal effects.",
        paste0("\"", col_nms[factor_cols], "\"", collapse = ", ")))
    }
    if (length(target_1based) == 0L)
      stop("All covariates are factors; marginal effects require at least one numeric covariate.",
           call. = FALSE)
  } else {
    target_1based <- sort(unique(as.integer(target_covariates)))
    if (any(target_1based < 1L) || any(target_1based > k))
      stop("`target_covariates` values must be in 1..ncol(data).", call. = FALSE)
    # Warn and drop user-requested factor columns
    requested_factors <- intersect(target_1based, factor_cols)
    if (length(requested_factors) > 0) {
      col_nms <- colnames(data)
      if (is.null(col_nms)) col_nms <- paste0("X", seq_len(k))
      warning(sprintf(
        "Factor/logical covariate(s) %s excluded from marginal effects.",
        paste0("\"", col_nms[requested_factors], "\"", collapse = ", ")),
        call. = FALSE)
      target_1based <- setdiff(target_1based, requested_factors)
    }
    if (length(target_1based) == 0L)
      stop("All requested `target_covariates` are factors; marginal effects require at least one numeric covariate.",
           call. = FALSE)
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

  # Step sizes for continuous covariates (bandwidth * sample SD)
  sds         <- apply(data, 2, sd)
  sds[sds == 0] <- 1.0   # guard against zero-variance columns
  h_vec       <- bandwidth * sds[target_1based]

  # Build the evaluation matrix
  X_eval <- switch(eval,
    "mean"     = data,
    "atmean"   = matrix(colMeans(data),          nrow = 1L),
    "atmedian" = matrix(apply(data, 2, median),  nrow = 1L)
  )

  # Delegate to C++ engine
  se_mat  <- NULL
  ci_lo   <- NULL
  ci_hi   <- NULL

  if (isTRUE(object$honesty)) {
    hd  <- object$honest_data
    res <- marginal_effects_honest_cpp(
      forest      = object$forest,
      X_eval      = X_eval,
      target_vars = target_0based,
      is_discrete = is_discrete_target,
      h_vec       = h_vec,
      Y_hon       = hd$Y_hon,
      n_hon       = hd$n_hon,
      M           = M,
      num_threads = resolve_num_threads(num.threads)
    )
    effects <- res$effects
    var_mat <- res$variance
    se_mat  <- sqrt(pmax(var_mat, 0))
    ci_lo   <- effects - 1.96 * se_mat
    ci_hi   <- effects + 1.96 * se_mat
  } else {
    effects <- marginal_effects_cpp(
      forest      = object$forest,
      X_eval      = X_eval,
      target_vars = target_0based,
      is_discrete = is_discrete_target,
      h_vec       = h_vec,
      M           = M,
      num_threads = resolve_num_threads(num.threads)
    )
  }

  # Attach names
  col_nms <- colnames(data)
  if (is.null(col_nms)) col_nms <- paste0("X", seq_len(k))
  rownames(effects) <- col_nms[target_1based]
  colnames(effects) <- paste0("P(Y=", seq_len(M), ")")
  if (!is.null(se_mat)) {
    rownames(se_mat) <- col_nms[target_1based]
    colnames(se_mat) <- paste0("P(Y=", seq_len(M), ")")
    rownames(ci_lo)  <- col_nms[target_1based]
    colnames(ci_lo)  <- paste0("P(Y=", seq_len(M), ")")
    rownames(ci_hi)  <- col_nms[target_1based]
    colnames(ci_hi)  <- paste0("P(Y=", seq_len(M), ")")
  }

  structure(
    list(
      effects           = effects,
      std.error         = se_mat,
      ci.lower          = ci_lo,
      ci.upper          = ci_hi,
      eval              = eval,
      target_covariates = target_1based,
      discrete_vars     = discrete_vars,
      bandwidth         = bandwidth,
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
  M <- ncol(x$effects)
  k <- nrow(x$effects)
  header <- switch(x$eval,
    "mean"     = "Average Marginal Effects",
    "atmean"   = "Marginal Effects at Mean",
    "atmedian" = "Marginal Effects at Median"
  )
  cat(sprintf("%s \u2014 jocf  [eval = \"%s\"]\n", header, x$eval))
  cat(sprintf("  %d covariate(s), %d outcome class(es)\n\n", k, M))
  mat <- x$effects
  if (is.null(rownames(mat))) rownames(mat) <- paste0("X", seq_len(k))
  print(round(mat, digits))
  if (!is.null(x$std.error)) {
    cat("\nStandard Errors:\n")
    print(round(x$std.error, digits))
    cat("\n95% Confidence Intervals (lower):\n")
    print(round(x$ci.lower, digits))
    cat("\n95% Confidence Intervals (upper):\n")
    print(round(x$ci.upper, digits))
  }
  invisible(x)
}
