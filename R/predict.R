#' Predict class probabilities and classifications from a fitted jocf object
#'
#' @param object A fitted object of class `"jocf"`.
#' @param newdata Matrix or data.frame with the same number of columns and
#'   column types as the training `X`.  Factor and logical columns are
#'   re-encoded using the level ordering stored at training time.  Unseen
#'   factor levels produce a warning and are mapped to the median code.
#' @param variance Logical. If `TRUE`, compute variance estimates and
#'   standard errors for each predicted probability.
#'   Requires `object$honesty == TRUE`. Default `FALSE`.
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
#'   \item{`variance`}{(nrow(newdata) x M) numeric matrix of variance
#'     estimates (only when `variance = TRUE`).}
#'   \item{`std.error`}{(nrow(newdata) x M) numeric matrix of standard
#'     errors, equal to `sqrt(pmax(variance, 0))` (only when
#'     `variance = TRUE`).}
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
#' \donttest{
#' ## Honest forest with variance estimation
#' fit_h <- jocf(Y, X, num.trees = 50, honesty = TRUE)
#' preds_h <- predict(fit_h, X_new, variance = TRUE)
#' head(preds_h$std.error)
#' }
#'
#' @export
predict.jocf <- function(object, newdata, variance = FALSE,
                          num.threads = NULL, ...) {
  if (!inherits(object, "jocf"))
    stop('`object` must be of class "jocf".', call. = FALSE)
  if (!is.logical(variance) || length(variance) != 1L)
    stop("`variance` must be TRUE or FALSE.", call. = FALSE)
  if (isTRUE(variance) && !isTRUE(object$honesty))
    stop("`variance = TRUE` requires an honest forest (honesty = TRUE).",
         call. = FALSE)

  # Encode factors using stored training-time metadata
  encoded <- encode_factors(newdata, factor_info = object$factor_info)
  newdata <- encoded$X_encoded
  if (ncol(newdata) != object$k)
    stop(sprintf("`newdata` must have %d column(s) (same as training X).",
                 object$k), call. = FALSE)
  if (anyNA(newdata))
    stop("Missing values are not allowed in `newdata`.", call. = FALSE)

  if (isTRUE(object$honesty) && isTRUE(variance)) {
    # Honest forest with variance estimation
    hd <- object$honest_data
    result <- predict_honest_cpp(
      forest           = object$forest,
      X_new            = newdata,
      Y_hon            = hd$Y_hon,
      n_hon            = hd$n_hon,
      M                = as.integer(object$M),
      compute_variance = TRUE,
      num_threads      = resolve_num_threads(num.threads)
    )
    predictions <- result$predictions
    votes       <- result$votes
    var_mat     <- result$variance

    class_prob <- apply(predictions, 1L, which.max)
    class_vote <- apply(votes, 1L, which.max)

    return(list(
      probabilities  = predictions,
      classification = list(prob = class_prob, vote = class_vote),
      variance       = var_mat,
      std.error      = sqrt(pmax(var_mat, 0))
    ))
  }

  # Standard prediction (non-honest or honest without variance)
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
