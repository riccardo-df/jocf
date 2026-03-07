# evaluation.R — Evaluation metrics for ordered outcome predictions
#
# Probability metrics: mean_squared_error, mean_absolute_error, mean_ranked_score
# Classification metrics: classification_error, mean_absolute_class_error, weighted_kappa

# --- Internal helpers --------------------------------------------------------

#' Build indicator matrix or pass through true-probability matrix
#'
#' @param y Integer vector of labels (1..M) or (n x M) matrix of true
#'   probabilities.
#' @param predictions (n x M) matrix of predicted probabilities.
#' @param use.true Logical. If `TRUE`, `y` is treated as an (n x M) matrix of
#'   true probabilities.
#' @return (n x M) matrix to compare against `predictions`.
#' @keywords internal
.resolve_y_matrix <- function(y, predictions, use.true) {
  M <- ncol(predictions)
  n <- nrow(predictions)

  if (use.true) {
    if (!is.matrix(y) || !is.numeric(y))
      stop("`y` must be a numeric matrix when `use.true = TRUE`.", call. = FALSE)
    if (nrow(y) != n || ncol(y) != M)
      stop("`y` must be an (n x M) matrix matching `predictions` dimensions.",
           call. = FALSE)
    return(y)
  }


  if (!is.numeric(y) || !is.null(dim(y)) && ncol(y) > 1L)
    stop("`y` must be an integer vector of labels when `use.true = FALSE`.",
         call. = FALSE)
  y <- as.integer(y)
  if (length(y) != n)
    stop("`y` and `predictions` must have the same number of observations.",
         call. = FALSE)
  if (any(y < 1L) || any(y > M))
    stop("`y` values must be in {1, ..., M}.", call. = FALSE)

  # Build indicator matrix
  ind <- matrix(0, nrow = n, ncol = M)
  ind[cbind(seq_len(n), y)] <- 1
  ind
}


# --- Probability metrics -----------------------------------------------------

#' Mean Squared Error (Multi-class Brier Score)
#'
#' Computes the mean over observations of \eqn{\sum_m (I(Y_i = m) - \hat p_m)^2}
#' (or the analogous expression with true probabilities when `use.true = TRUE`).
#'
#' @param y Integer vector with values in `{1, ..., M}`, or an (n x M) matrix
#'   of true probabilities when `use.true = TRUE`.
#' @param predictions (n x M) numeric matrix of predicted probabilities.
#' @param use.true Logical.  If `TRUE`, `y` is interpreted as a matrix of true
#'   conditional probabilities rather than observed labels.  Default `FALSE`.
#'
#' @return A single numeric value.
#'
#' @examples
#' set.seed(1)
#' Y <- sample(1:3, 50, replace = TRUE)
#' P <- matrix(1/3, nrow = 50, ncol = 3)
#' mean_squared_error(Y, P)
#'
#' @export
mean_squared_error <- function(y, predictions, use.true = FALSE) {
  predictions <- as.matrix(predictions)
  y_mat <- .resolve_y_matrix(y, predictions, use.true)
  mean(rowSums((y_mat - predictions)^2))
}


#' Mean Absolute Error
#'
#' Computes the mean over observations of \eqn{\sum_m |I(Y_i = m) - \hat p_m|}
#' (or the analogous expression with true probabilities when `use.true = TRUE`).
#'
#' @inheritParams mean_squared_error
#'
#' @return A single numeric value.
#'
#' @examples
#' set.seed(1)
#' Y <- sample(1:3, 50, replace = TRUE)
#' P <- matrix(1/3, nrow = 50, ncol = 3)
#' mean_absolute_error(Y, P)
#'
#' @export
mean_absolute_error <- function(y, predictions, use.true = FALSE) {
  predictions <- as.matrix(predictions)
  y_mat <- .resolve_y_matrix(y, predictions, use.true)
  mean(rowSums(abs(y_mat - predictions)))
}


#' Mean Ranked Probability Score (RPS)
#'
#' Computes the Ranked Probability Score, which is sensitive to the ordering of
#' classes.  Defined as
#' \deqn{\textrm{RPS} = \frac{1}{M-1} \cdot \frac{1}{n}
#'   \sum_i \sum_{m=1}^{M-1} \bigl(F_{\textrm{pred}}(m) - F_{\textrm{true}}(m)\bigr)^2}
#' where \eqn{F(m) = \sum_{j=1}^{m} p_j} is the CDF.
#'
#' @inheritParams mean_squared_error
#'
#' @return A single numeric value.
#'
#' @examples
#' set.seed(1)
#' Y <- sample(1:3, 50, replace = TRUE)
#' P <- matrix(1/3, nrow = 50, ncol = 3)
#' mean_ranked_score(Y, P)
#'
#' @export
mean_ranked_score <- function(y, predictions, use.true = FALSE) {
  predictions <- as.matrix(predictions)
  M <- ncol(predictions)
  y_mat <- .resolve_y_matrix(y, predictions, use.true)

  # Cumulative sums along classes (n x M matrices)
  cdf_pred <- t(apply(predictions, 1, cumsum))
  cdf_true <- t(apply(y_mat, 1, cumsum))

  # RPS uses only M-1 CDF values (m = 1, ..., M-1)
  rps_per_obs <- rowSums((cdf_pred[, -M, drop = FALSE] -
                           cdf_true[, -M, drop = FALSE])^2)
  mean(rps_per_obs) / (M - 1)
}


# --- Classification metrics --------------------------------------------------

#' Classification Error (Misclassification Rate)
#'
#' Computes the proportion of observations where the predicted class does not
#' match the true class.
#'
#' @param y Integer vector of true class labels.
#' @param predictions Integer vector of predicted class labels (same length as
#'   `y`).
#'
#' @return A single numeric value in \eqn{[0, 1]}.
#'
#' @examples
#' y    <- c(1L, 2L, 3L, 1L, 2L)
#' yhat <- c(1L, 2L, 2L, 1L, 3L)
#' classification_error(y, yhat)
#'
#' @export
classification_error <- function(y, predictions) {
  y <- as.integer(y)
  predictions <- as.integer(predictions)
  if (length(y) != length(predictions))
    stop("`y` and `predictions` must have the same length.", call. = FALSE)
  mean(y != predictions)
}


#' Mean Absolute Class Error
#'
#' Computes \eqn{\frac{1}{n} \sum_i |Y_i - \hat Y_i|}.  Unlike plain
#' misclassification rate, this exploits the ordering of classes: predicting
#' class 1 when the truth is 5 is penalised more than predicting class 4.
#'
#' @inheritParams classification_error
#'
#' @return A single non-negative numeric value.
#'
#' @examples
#' y    <- c(1L, 2L, 3L, 1L, 2L)
#' yhat <- c(1L, 2L, 2L, 1L, 3L)
#' mean_absolute_class_error(y, yhat)
#'
#' @export
mean_absolute_class_error <- function(y, predictions) {
  y <- as.integer(y)
  predictions <- as.integer(predictions)
  if (length(y) != length(predictions))
    stop("`y` and `predictions` must have the same length.", call. = FALSE)
  mean(abs(y - predictions))
}


#' Weighted Cohen's Kappa
#'
#' Computes Cohen's weighted kappa for ordered classifications.  The weight
#' matrix entry \eqn{W_{ij}} measures the disagreement between classes \eqn{i}
#' and \eqn{j}:
#' \itemize{
#'   \item **Quadratic** (default): \eqn{W_{ij} = (i - j)^2 / (M - 1)^2}
#'   \item **Linear**: \eqn{W_{ij} = |i - j| / (M - 1)}
#' }
#'
#' @inheritParams classification_error
#' @param M Integer.  Number of outcome classes.  Required because not all
#'   classes may be observed in `y` or `predictions`.
#' @param type Character.  `"quadratic"` (default) or `"linear"`.
#'
#' @return A single numeric value, or `NA` if the denominator is zero (e.g.,
#'   when one rater is constant).
#'
#' @examples
#' y    <- c(1L, 2L, 3L, 1L, 2L)
#' yhat <- c(1L, 2L, 2L, 1L, 3L)
#' weighted_kappa(y, yhat, M = 3)
#'
#' @export
weighted_kappa <- function(y, predictions, M, type = "quadratic") {
  y <- as.integer(y)
  predictions <- as.integer(predictions)
  M <- as.integer(M)

  if (length(y) != length(predictions))
    stop("`y` and `predictions` must have the same length.", call. = FALSE)
  if (!is.numeric(M) || length(M) != 1L || M < 2L)
    stop("`M` must be an integer >= 2.", call. = FALSE)
  if (!type %in% c("quadratic", "linear"))
    stop('`type` must be "quadratic" or "linear".', call. = FALSE)

  n <- length(y)

  # Weight matrix
  grid <- expand.grid(i = seq_len(M), j = seq_len(M))
  if (type == "quadratic") {
    W <- matrix((grid$i - grid$j)^2 / (M - 1)^2, M, M)
  } else {
    W <- matrix(abs(grid$i - grid$j) / (M - 1), M, M)
  }

  # Observed confusion matrix
  O <- matrix(0L, M, M)
  for (idx in seq_len(n)) {
    O[y[idx], predictions[idx]] <- O[y[idx], predictions[idx]] + 1L
  }
  O <- O / n

  # Expected matrix under independence
  hist_y <- tabulate(y, nbins = M) / n
  hist_p <- tabulate(predictions, nbins = M) / n
  E <- outer(hist_y, hist_p)

  num   <- sum(W * O)
  denom <- sum(W * E)

  if (denom == 0) return(NA_real_)
  1 - num / denom
}
