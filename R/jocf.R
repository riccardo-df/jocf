#' Fit a Joint Ordered Correlation Forest
#'
#' Grows a single random forest whose splitting criterion jointly minimises
#' average estimation error across all M outcome classes simultaneously.
#' The criterion is equivalent to average CART Gini impurity (Proposition 1
#' in the package vignette).
#'
#' @param Y Integer vector with values in `{1, ..., M}` giving the ordered
#'   discrete outcome.
#' @param X Numeric matrix or data.frame of covariates (n x k).
#' @param num.trees Positive integer. Number of trees. Default `2000`.
#' @param min.node.size Positive integer. Minimum number of observations in
#'   a terminal node. Default `5`.
#' @param max.depth Positive integer or `NULL`. Maximum tree depth. `NULL`
#'   (default) grows trees without depth constraint. `max.depth = 1` produces
#'   stumps (a single split).
#' @param mtry Positive integer or `NULL`. Number of features to consider at
#'   each split. Default `NULL` uses `floor(sqrt(ncol(X)))`.
#' @param splitting.rule Character, one of `"simple"` (default) or
#'   `"weighted"`. `"simple"` uses equal weights across classes;
#'   `"weighted"` standardises each class by its population-level Bernoulli
#'   variance (see `vignette("jocf")`).
#' @param honesty Logical. Must be `FALSE`; `TRUE` is not yet implemented.
#' @param num.threads Positive integer or `NULL`. Number of OpenMP threads to
#'   use when growing trees and computing in-sample predictions. `NULL`
#'   (default) uses all available cores.
#' @param ... Currently unused.
#'
#' @return An object of class `"jocf"` — a list with components:
#' \describe{
#'   \item{`predictions`}{(n x M) matrix of in-sample predicted class
#'     probabilities. Rows sum to 1 and are non-negative.}
#'   \item{`classification`}{Named list with two integer vectors (values in
#'     `1, ..., M`):
#'     \describe{
#'       \item{`prob`}{Probability-based classification: argmax of forest-
#'         averaged probabilities (see `vignette("jocf-theory")`).}
#'       \item{`vote`}{Majority-vote classification: each tree votes for its
#'         leaf argmax, then the class with the most votes wins.  Unique to
#'         the unified OCF (see `vignette("jocf-theory")`).}
#'     }}
#'   \item{`forest`}{List of B tree structures; used by [predict.jocf()].}
#'   \item{`splitting.rule`}{Character; the splitting rule used.}
#'   \item{`M`}{Integer; number of outcome classes.}
#'   \item{`num.trees`}{Integer; number of trees grown.}
#'   \item{`k`}{Integer; number of training covariates.}
#'   \item{`call`}{The matched call.}
#' }
#'
#' @examples
#' ## Simulate ordered outcome data
#' set.seed(42)
#' n <- 150
#' X <- matrix(rnorm(n * 3), ncol = 3)
#' Y <- sample(1:3, n, replace = TRUE, prob = c(0.3, 0.5, 0.2))
#'
#' ## Fit a joint ordered correlation forest
#' fit <- jocf(Y, X, num.trees = 50)
#'
#' ## In-sample predicted probabilities (n x M matrix)
#' head(fit$predictions)
#'
#' ## Classifications
#' table(fit$classification$prob)   # probability-based
#' table(fit$classification$vote)   # majority-vote
#'
#' @export
jocf <- function(Y,
                 X,
                 num.trees      = 2000L,
                 min.node.size  = 5L,
                 max.depth      = NULL,
                 mtry           = NULL,
                 splitting.rule = "simple",
                 honesty        = FALSE,
                 num.threads    = NULL,
                 ...) {
  cl <- match.call()

  Y <- as.integer(Y)
  X <- as.matrix(X)
  validate_jocf_inputs(Y, X, num.trees, min.node.size, max.depth, mtry,
                       splitting.rule, honesty)

  M         <- max(Y)
  n         <- length(Y)
  k         <- ncol(X)
  mtry      <- if (is.null(mtry)) floor(sqrt(k)) else as.integer(mtry)
  max_depth <- if (is.null(max.depth)) -1L else as.integer(max.depth)

  lambda <- if (splitting.rule == "weighted") {
    compute_lambda(Y, M)
  } else {
    rep(1.0, M)
  }

  forest_result <- grow_forest_cpp(
    Y             = Y,
    X             = X,
    num_trees     = as.integer(num.trees),
    min_node_size = as.integer(min.node.size),
    max_depth     = max_depth,
    mtry          = as.integer(mtry),
    M             = as.integer(M),
    lambda        = lambda,
    num_threads   = resolve_num_threads(num.threads)
  )

  predictions <- forest_result$predictions
  votes       <- forest_result$votes

  # Probability-based classification: argmax of forest-averaged probabilities
  class_prob <- apply(predictions, 1L, which.max)
  # Majority-vote classification: argmax of per-tree vote counts
  class_vote <- apply(votes, 1L, which.max)

  structure(
    list(
      predictions    = predictions,
      classification = list(prob = class_prob, vote = class_vote),
      forest         = forest_result$forest,
      splitting.rule = splitting.rule,
      max.depth      = max.depth,
      M              = M,
      num.trees      = as.integer(num.trees),
      k              = k,
      call           = cl
    ),
    class = "jocf"
  )
}
