#' Predict class probabilities from a fitted jocf object
#'
#' @param object A fitted object of class `"jocf"`.
#' @param newdata Numeric matrix or data.frame with the same number of
#'   columns as the training `X`.
#' @param num.threads Positive integer or `NULL`. Number of OpenMP threads.
#'   `NULL` (default) uses all available cores.
#' @param ... Currently unused.
#'
#' @return An (nrow(newdata) x M) numeric matrix of predicted class
#'   probabilities. Rows sum to 1 and are non-negative.
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

  predict_forest_cpp(object$forest, newdata, as.integer(object$M),
                     resolve_num_threads(num.threads))
}
