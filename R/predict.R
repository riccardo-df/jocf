#' Predict class probabilities and classifications from a fitted jocf object
#'
#' @param object A fitted object of class `"jocf"`.
#' @param newdata Numeric matrix or data.frame with the same number of
#'   columns as the training `X`.
#' @param num.threads Positive integer or `NULL`. Number of OpenMP threads.
#'   `NULL` (default) uses all available cores.
#' @param ... Currently unused.
#'
#' @return A named list with components:
#' \describe{
#'   \item{`probabilities`}{(nrow(newdata) x M) numeric matrix of predicted
#'     class probabilities. Rows sum to 1 and are non-negative.}
#'   \item{`classification`}{Named list with two integer vectors (values in
#'     `1, ..., M`):
#'     \describe{
#'       \item{`prob`}{Probability-based classification: argmax of forest-
#'         averaged probabilities (see `vignette("jocf-theory")`).}
#'       \item{`vote`}{Majority-vote classification: each tree votes for its
#'         leaf argmax, then the class with the most votes wins.  Unique to
#'         the unified OCF (see `vignette("jocf-theory")`).}
#'     }}
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
#' ## Predict on new observations
#' X_new <- matrix(rnorm(10 * 3), ncol = 3)
#' preds <- predict(fit, X_new)
#' head(preds$probabilities)
#' preds$classification$prob
#' preds$classification$vote
#'
#' @export
predict.jocf <- function(object, newdata, num.threads = NULL, ...) {
  if (!inherits(object, "jocf"))
    stop('`object` must be of class "jocf".', call. = FALSE)

  newdata <- as.matrix(newdata)
  if (!is.numeric(newdata))
    stop("`newdata` must be numeric.", call. = FALSE)
  if (ncol(newdata) != object$k)
    stop(sprintf("`newdata` must have %d column(s) (same as training X).",
                 object$k), call. = FALSE)
  if (anyNA(newdata))
    stop("Missing values are not allowed in `newdata`.", call. = FALSE)

  result <- predict_forest_cpp(object$forest, newdata, as.integer(object$M),
                               resolve_num_threads(num.threads))

  predictions <- result$predictions
  votes       <- result$votes

  class_prob <- apply(predictions, 1L, which.max)
  class_vote <- apply(votes, 1L, which.max)

  list(
    probabilities  = predictions,
    classification = list(prob = class_prob, vote = class_vote)
  )
}
