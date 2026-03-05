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
#'   \item{`forest`}{List of B tree structures; used by [predict.jocf()].}
#'   \item{`splitting.rule`}{Character; the splitting rule used.}
#'   \item{`M`}{Integer; number of outcome classes.}
#'   \item{`num.trees`}{Integer; number of trees grown.}
#'   \item{`k`}{Integer; number of training covariates.}
#'   \item{`call`}{The matched call.}
#' }
#'
#' @export
jocf <- function(Y,
                 X,
                 num.trees      = 2000L,
                 min.node.size  = 5L,
                 mtry           = NULL,
                 splitting.rule = "simple",
                 honesty        = FALSE,
                 num.threads    = NULL,
                 ...) {
  cl <- match.call()

  Y <- as.integer(Y)
  X <- as.matrix(X)
  validate_jocf_inputs(Y, X, num.trees, min.node.size, mtry,
                       splitting.rule, honesty)

  M    <- max(Y)
  n    <- length(Y)
  k    <- ncol(X)
  mtry <- if (is.null(mtry)) floor(sqrt(k)) else as.integer(mtry)

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
    mtry          = as.integer(mtry),
    M             = as.integer(M),
    lambda        = lambda,
    num_threads   = resolve_num_threads(num.threads)
  )

  structure(
    list(
      predictions    = forest_result$predictions,
      forest         = forest_result$forest,
      splitting.rule = splitting.rule,
      M              = M,
      num.trees      = as.integer(num.trees),
      k              = k,
      call           = cl
    ),
    class = "jocf"
  )
}
