#' Fit a Joint Ordered Correlation Forest
#'
#' Grows a single random forest whose splitting criterion jointly minimises
#' average estimation error across all M outcome classes simultaneously.
#' The criterion is equivalent to average CART Gini impurity (Proposition 1
#' in the package vignette).
#'
#' @param Y Integer vector with values in `{1, ..., M}` giving the ordered
#'   discrete outcome.
#' @param X Matrix or data.frame of covariates (n x k). Columns may be numeric,
#'   factor (ordered or unordered), or logical.  Unordered factor levels are
#'   internally sorted by `mean(Y)` and encoded as integer codes; ordered factors
#'   use their existing level order.  Character columns are not accepted; convert
#'   to factor first.
#' @param num.trees Positive integer. Number of trees. Default `2000`.
#' @param min.node.size Positive integer. Minimum number of observations in
#'   a terminal node. Default `5`.
#' @param max.depth Positive integer or `NULL`. Maximum tree depth. `NULL`
#'   (default) grows trees without depth constraint. `max.depth = 1` produces
#'   stumps (a single split).
#' @param sample.fraction Numeric in `(0, 1]`. Fraction of observations to
#'   draw **without replacement** for each tree. Default `0.5`.
#' @param mtry Positive integer or `NULL`. Number of features to consider at
#'   each split. Default `NULL` uses `floor(sqrt(ncol(X)))`.
#' @param splitting.rule Character, one of `"simple"` (default) or
#'   `"weighted"`. `"simple"` uses equal weights across classes;
#'   `"weighted"` standardises each class by its population-level Bernoulli
#'   variance (see `vignette("jocf")`).
#' @param tune.parameters Character. `"none"` (default) disables tuning.
#'   `"all"` tunes `mtry`, `min.node.size`, and `sample.fraction`.
#'   Alternatively, a character vector naming which parameters to tune
#'   (any subset of `c("mtry", "min.node.size", "sample.fraction")`).
#'   Tuning follows the GRF approach: many small mini-forests are evaluated
#'   at random parameter draws, a Kriging surrogate is fitted, and the
#'   parameters with the lowest predicted debiased OOB error are selected
#'   before growing the full forest.  Requires the \pkg{DiceKriging} package.
#' @param tune.num.trees Positive integer. Number of trees per mini-forest
#'   during tuning. Default `50`.
#' @param tune.num.reps Positive integer. Number of random parameter draws
#'   to evaluate with mini-forests. Default `100`.
#' @param tune.num.draws Positive integer. Number of new random candidates
#'   evaluated via the Kriging surrogate. Default `1000`.
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
#'   \item{`sample.fraction`}{Numeric; the subsample fraction used.}
#'   \item{`M`}{Integer; number of outcome classes.}
#'   \item{`num.trees`}{Integer; number of trees grown.}
#'   \item{`k`}{Integer; number of training covariates.}
#'   \item{`tuning.output`}{`NULL` when tuning is off, otherwise a named list
#'     with components `status` (`"tuned"`, `"default"`, or `"failure"`),
#'     `params` (selected parameter values), `error` (debiased OOB error),
#'     and `grid` (data frame of evaluated draws and errors).}
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
#' ## Factor covariates
#' X_mixed <- data.frame(x1 = rnorm(n), x2 = factor(sample(c("A","B"), n, TRUE)))
#' fit2 <- jocf(Y, X_mixed, num.trees = 50)
#' head(fit2$predictions)
#'
#' \donttest{
#' ## Built-in hyperparameter tuning (requires DiceKriging)
#' if (requireNamespace("DiceKriging", quietly = TRUE)) {
#'   fit_tuned <- jocf(Y, X, num.trees = 50,
#'                     tune.parameters = "all",
#'                     tune.num.trees = 10,
#'                     tune.num.reps = 20,
#'                     tune.num.draws = 50)
#'   fit_tuned$tuning.output$params
#' }
#' }
#'
#' @export
jocf <- function(Y,
                 X,
                 num.trees        = 2000L,
                 min.node.size    = 5L,
                 max.depth        = NULL,
                 sample.fraction  = 0.5,
                 mtry             = NULL,
                 splitting.rule   = "simple",
                 tune.parameters  = "none",
                 tune.num.trees   = 50L,
                 tune.num.reps    = 100L,
                 tune.num.draws   = 1000L,
                 honesty          = FALSE,
                 num.threads      = NULL,
                 ...) {
  cl <- match.call()

  Y <- as.integer(Y)
  validate_jocf_inputs(Y, X, num.trees, min.node.size, max.depth,
                       sample.fraction, mtry, splitting.rule, honesty)

  # Encode factor / ordered / logical columns to numeric codes
  encoded <- encode_factors(X, Y = Y)
  X_mat   <- encoded$X_encoded
  fi      <- encoded$factor_info

  M         <- max(Y)
  n         <- length(Y)
  k         <- ncol(X_mat)
  mtry      <- if (is.null(mtry)) floor(sqrt(k)) else as.integer(mtry)
  max_depth <- if (is.null(max.depth)) -1L else as.integer(max.depth)
  n_sub     <- as.integer(ceiling(sample.fraction * n))

  lambda <- if (splitting.rule == "weighted") {
    compute_lambda(Y, M)
  } else {
    rep(1.0, M)
  }

  # Hyperparameter tuning
  tune_params <- validate_tune_inputs(tune.parameters, tune.num.trees,
                                      tune.num.reps, tune.num.draws)
  tuning_output <- NULL

  if (length(tune_params) > 0L) {
    defaults <- list(
      mtry            = as.integer(mtry),
      min.node.size   = as.integer(min.node.size),
      sample.fraction = sample.fraction
    )
    tuning_output <- tune_jocf(
      Y              = Y,
      X_mat          = X_mat,
      M              = M,
      tune_params    = tune_params,
      defaults       = defaults,
      tune.num.trees = as.integer(tune.num.trees),
      tune.num.reps  = as.integer(tune.num.reps),
      tune.num.draws = as.integer(tune.num.draws),
      max_depth      = max_depth,
      lambda         = lambda,
      num_threads    = resolve_num_threads(num.threads)
    )
    # Override parameters with tuned values
    mtry            <- as.integer(tuning_output$params$mtry)
    min.node.size   <- as.integer(tuning_output$params$min.node.size)
    sample.fraction <- tuning_output$params$sample.fraction
    n_sub           <- as.integer(ceiling(sample.fraction * n))
  }

  forest_result <- grow_forest_cpp(
    Y             = Y,
    X             = X_mat,
    num_trees     = as.integer(num.trees),
    min_node_size = as.integer(min.node.size),
    max_depth     = max_depth,
    n_sub         = n_sub,
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
      forest          = forest_result$forest,
      splitting.rule  = splitting.rule,
      sample.fraction = sample.fraction,
      max.depth       = max.depth,
      M              = M,
      num.trees      = as.integer(num.trees),
      k              = k,
      factor_info     = fi,
      col_names       = colnames(X_mat),
      tuning.output   = tuning_output,
      call           = cl
    ),
    class = "jocf"
  )
}
