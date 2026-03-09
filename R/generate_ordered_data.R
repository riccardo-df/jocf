#' Generate Ordered Outcome Data from an Ordered Logit DGP
#'
#' Simulates data from an ordered logistic model with six covariates and
#' configurable class proportions.  The DGP matches the one shipped with the
#' `ocf` package when called with default arguments, and extends it by
#' allowing control over the marginal class distribution via
#' `marginal_probs`.
#'
#' The data-generating process is:
#' \enumerate{
#'   \item **Covariates** (k = 6): \eqn{x_1, x_3 \sim N(0,1)};
#'     \eqn{x_2, x_4 \sim \textrm{Bernoulli}(0.4)};
#'     \eqn{x_5, x_6} are noise covariates drawn from the same distributions
#'     with zero coefficients.
#'   \item **Linear index**: \eqn{g(X) = x_1 + x_2 + 0.5 x_3 + 0.5 x_4}.
#'   \item **Latent outcome**: \eqn{Y^* = g(X) + \varepsilon},
#'     \eqn{\varepsilon \sim \textrm{Logistic}(0,1)}.
#'   \item **Discretisation**: \eqn{Y = m} when
#'     \eqn{\zeta_{m-1} < Y^* \le \zeta_m}, where the cutpoints
#'     \eqn{\zeta} are chosen so that the population marginal class
#'     proportions match `marginal_probs`.
#' }
#'
#' @param n Positive integer.  Number of observations to draw.
#' @param n_categories Positive integer (>= 2).  Number of outcome classes.
#'   Default `3`.
#' @param marginal_probs Numeric vector of target marginal class proportions,
#'   or `NULL` (default) for equal proportions
#'   `rep(1/n_categories, n_categories)`.  Must be positive and sum to 1.
#'   If supplied, its length must equal `n_categories`.
#'
#' @return A list with components:
#' \describe{
#'   \item{`sample`}{(n x 7) data.frame with columns `Y`, `x1`, ..., `x6`.}
#'   \item{`true_probs`}{(n x M) matrix of true conditional probabilities
#'     \eqn{P(Y = m \mid X_i)} from the ordered logit model.}
#'   \item{`marginal_probs`}{The (possibly defaulted) probability vector
#'     used to set the cutpoints.}
#'   \item{`true_me_atmean`}{(6 x M) matrix of true marginal effects evaluated
#'     at \code{colMeans(X)}.  Continuous covariates use the analytic derivative;
#'     discrete covariates use the unit-difference
#'     \eqn{P(Y=m \mid x_j=1) - P(Y=m \mid x_j=0)}.}
#'   \item{`true_me_atmedian`}{Same as `true_me_atmean` but evaluated at
#'     column medians of \code{X}.}
#' }
#'
#' @examples
#' ## Default: balanced M = 3
#' d <- generate_ordered_data(500)
#' table(d$sample$Y)
#' head(d$true_probs)
#'
#' ## Imbalanced classes
#' d2 <- generate_ordered_data(500, marginal_probs = c(0.7, 0.2, 0.1))
#' table(d2$sample$Y)
#'
#' ## Four outcome classes
#' d3 <- generate_ordered_data(500, n_categories = 4)
#' table(d3$sample$Y)
#'
#' @importFrom stats plogis dlogis rnorm rbinom rlogis quantile median
#' @export
generate_ordered_data <- function(n,
                                  n_categories   = 3L,
                                  marginal_probs = NULL) {


  ## --- Input validation ---------------------------------------------------
  if (!is.numeric(n) || length(n) != 1L || n < 1L || n != floor(n))
    stop("`n` must be a positive integer.", call. = FALSE)
  n <- as.integer(n)

  if (!is.numeric(n_categories) || length(n_categories) != 1L ||
      n_categories < 2L || n_categories != floor(n_categories))
    stop("`n_categories` must be an integer >= 2.", call. = FALSE)
  n_categories <- as.integer(n_categories)

  if (is.null(marginal_probs)) {
    marginal_probs <- rep(1 / n_categories, n_categories)
  } else {
    if (!is.numeric(marginal_probs) || any(marginal_probs <= 0))
      stop("`marginal_probs` must be a positive numeric vector.", call. = FALSE)
    if (abs(sum(marginal_probs) - 1) > 1e-8)
      stop("`marginal_probs` must sum to 1.", call. = FALSE)
    if (length(marginal_probs) != n_categories)
      stop("`marginal_probs` must have length equal to `n_categories`.",
           call. = FALSE)
  }

  M <- n_categories

  ## --- Compute cutpoints from a large population --------------------------
  n_pop   <- 1e6L
  x1_pop  <- rnorm(n_pop)
  x2_pop  <- rbinom(n_pop, 1L, 0.4)
  x3_pop  <- rnorm(n_pop)
  x4_pop  <- rbinom(n_pop, 1L, 0.4)
  g_pop   <- x1_pop + x2_pop + 0.5 * x3_pop + 0.5 * x4_pop
  eps_pop <- rlogis(n_pop)
  ystar_pop <- g_pop + eps_pop

  cum_probs <- cumsum(marginal_probs)
  # M-1 cutpoints: quantiles of Y* at cumulative probabilities
  cutpoints <- quantile(ystar_pop, probs = cum_probs[-M], names = FALSE)

  ## --- Draw sample --------------------------------------------------------
  x1 <- rnorm(n)
  x2 <- rbinom(n, 1L, 0.4)
  x3 <- rnorm(n)
  x4 <- rbinom(n, 1L, 0.4)
  x5 <- rnorm(n)
  x6 <- rbinom(n, 1L, 0.4)
  g  <- x1 + x2 + 0.5 * x3 + 0.5 * x4
  eps <- rlogis(n)
  ystar <- g + eps

  # Discretise: Y = 1 if ystar <= cutpoints[1], ..., Y = M if ystar > cutpoints[M-1]
  Y <- findInterval(ystar, cutpoints) + 1L

  sample_df <- data.frame(Y = Y, x1 = x1, x2 = x2, x3 = x3,
                           x4 = x4, x5 = x5, x6 = x6)

  ## --- True conditional probabilities -------------------------------------
  # P(Y <= m | X) = plogis(cutpoints[m] - g(X))
  # P(Y = m | X) = P(Y <= m | X) - P(Y <= m-1 | X)
  true_probs <- matrix(0, nrow = n, ncol = M)
  cum_prob_prev <- rep(0, n)
  for (m in seq_len(M)) {
    if (m < M) {
      cum_prob_cur <- plogis(cutpoints[m] - g)
    } else {
      cum_prob_cur <- rep(1, n)
    }
    true_probs[, m] <- cum_prob_cur - cum_prob_prev
    cum_prob_prev <- cum_prob_cur
  }

  ## --- True marginal effects ------------------------------------------------
  beta <- c(1, 1, 0.5, 0.5, 0, 0)
  is_discrete <- c(FALSE, TRUE, FALSE, TRUE, FALSE, TRUE)  # x2, x4, x6
  cov_names <- paste0("x", 1:6)
  class_names <- paste0("Y", seq_len(M))
  zeta <- c(-Inf, cutpoints, Inf)   # length M+1

  X_matrix <- as.matrix(sample_df[, cov_names])
  x_mean   <- colMeans(X_matrix)
  x_median <- apply(X_matrix, 2, median)

  compute_true_me <- function(x_eval) {
    me <- matrix(0, nrow = 6, ncol = M)
    g_eval <- sum(x_eval * beta)
    for (j in seq_len(6)) {
      if (is_discrete[j]) {
        # Unit difference: P(Y=m | x_j=1, rest) - P(Y=m | x_j=0, rest)
        g1 <- g_eval - x_eval[j] * beta[j] + 1 * beta[j]
        g0 <- g_eval - x_eval[j] * beta[j] + 0 * beta[j]
        for (m in seq_len(M)) {
          p1 <- plogis(zeta[m + 1] - g1) - plogis(zeta[m] - g1)
          p0 <- plogis(zeta[m + 1] - g0) - plogis(zeta[m] - g0)
          me[j, m] <- p1 - p0
        }
      } else {
        # Analytic derivative: [dlogis(zeta_{m-1} - g) - dlogis(zeta_m - g)] * beta_j
        for (m in seq_len(M)) {
          me[j, m] <- (dlogis(zeta[m] - g_eval) -
                        dlogis(zeta[m + 1] - g_eval)) * beta[j]
        }
      }
    }
    rownames(me) <- cov_names
    colnames(me) <- class_names
    me
  }

  true_me_atmean   <- compute_true_me(x_mean)
  true_me_atmedian <- compute_true_me(x_median)

  list(
    sample           = sample_df,
    true_probs       = true_probs,
    marginal_probs   = marginal_probs,
    true_me_atmean   = true_me_atmean,
    true_me_atmedian = true_me_atmedian
  )
}
