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
#' @param honesty Logical. If `TRUE`, the forest uses sample splitting: trees
#'   are grown on a training subsample and leaf predictions are computed from a
#'   held-out honesty subsample. This enables valid variance estimation via
#'   [predict.jocf()] and [marginal_effects.jocf()]. Default `FALSE`.
#' @param honesty.fraction Numeric in `(0, 1)`. Fraction of observations
#'   allocated to the honesty sample when `honesty = TRUE`. Default `0.5`.
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
#'   \item{`honesty`}{Logical; whether honesty was used.}
#'   \item{`honesty.fraction`}{Numeric fraction allocated to honesty sample
#'     (`NULL` when non-honest).}
#'   \item{`honest_data`}{Internal list with honesty metadata for
#'     variance estimation (`NULL` when non-honest).}
#'   \item{`M`}{Integer; number of outcome classes.}
#'   \item{`num.trees`}{Integer; number of trees grown.}
#'   \item{`k`}{Integer; number of training covariates.}
#'   \item{`X_train`}{Factor-encoded numeric training matrix (n x k); used as
#'     the default `data` argument by [marginal_effects.jocf()].}
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
#' ## Honest forest with variance estimation
#' fit_h <- jocf(Y, X, num.trees = 50, honesty = TRUE)
#' fit_h$honesty
#' X_new <- matrix(rnorm(5 * 3), ncol = 3)
#' pred_h <- predict(fit_h, X_new, variance = TRUE)
#' head(pred_h$std.error)
#'
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
                 honesty.fraction = 0.5,
                 num.threads      = NULL,
                 ...) {
  cl <- match.call()

  Y <- as.integer(Y)
  validate_jocf_inputs(Y, X, num.trees, min.node.size, max.depth,
                       sample.fraction, mtry, splitting.rule, honesty,
                       honesty.fraction)

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

  honest_data <- NULL

  if (isTRUE(honesty)) {
    # Sample splitting: first n_hon obs → honesty, rest → training
    perm <- sample.int(n)
    n_hon <- as.integer(floor(honesty.fraction * n))
    n_tr  <- n - n_hon
    hon_indices <- perm[seq_len(n_hon)] - 1L          # 0-based
    tr_indices  <- perm[seq(n_hon + 1L, n)] - 1L      # 0-based
    n_sub_tr    <- as.integer(ceiling(sample.fraction * n_tr))

    forest_result <- grow_forest_honest_cpp(
      Y             = Y,
      X             = X_mat,
      num_trees     = as.integer(num.trees),
      min_node_size = as.integer(min.node.size),
      max_depth     = max_depth,
      n_sub_tr      = n_sub_tr,
      mtry          = as.integer(mtry),
      M             = as.integer(M),
      lambda        = lambda,
      tr_indices    = tr_indices,
      hon_indices   = hon_indices,
      num_threads   = resolve_num_threads(num.threads)
    )

    # Store honesty metadata for predict-time variance estimation
    Y_hon_0 <- as.integer(Y[hon_indices + 1L] - 1L)   # 0-indexed class labels
    honest_data <- list(
      Y_hon       = Y_hon_0,
      n_hon       = n_hon,
      hon_indices = hon_indices
    )
  } else {
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
  }

  predictions <- forest_result$predictions
  votes       <- forest_result$votes

  # Probability-based classification: argmax of forest-averaged probabilities
  class_prob <- apply(predictions, 1L, which.max)
  # Majority-vote classification: argmax of per-tree vote counts
  class_vote <- apply(votes, 1L, which.max)

  structure(
    list(
      predictions     = predictions,
      classification  = list(prob = class_prob, vote = class_vote),
      forest          = forest_result$forest,
      splitting.rule  = splitting.rule,
      sample.fraction = sample.fraction,
      max.depth       = max.depth,
      honesty         = honesty,
      honesty.fraction = if (isTRUE(honesty)) honesty.fraction else NULL,
      honest_data     = honest_data,
      M               = M,
      num.trees       = as.integer(num.trees),
      n               = n,
      k               = k,
      factor_info     = fi,
      col_names       = colnames(X_mat),
      X_train         = X_mat,
      tuning.output   = tuning_output,
      call            = cl
    ),
    class = "jocf"
  )
}


#' Print method for jocf objects
#'
#' Displays a concise one-screen overview of a fitted joint ordered correlation
#' forest, following the ranger-style convention.
#'
#' @param x A `"jocf"` object.
#' @param ... Currently unused.
#'
#' @return Invisibly returns `x`.
#' @export
print.jocf <- function(x, ...) {
  cat("Joint Ordered Correlation Forest\n\n")
  cat("Call:", deparse(x$call), "\n\n")

  n_obs <- if (!is.null(x$n)) x$n else nrow(x$predictions)
  cat(sprintf("  %-20s%s\n", "Type:", "Joint ordered correlation forest"))
  cat(sprintf("  %-20s%d\n", "Number of trees:", x$num.trees))
  cat(sprintf("  %-20s%d\n", "Observations:", n_obs))
  cat(sprintf("  %-20s%d\n", "Covariates:", x$k))
  cat(sprintf("  %-20s%d\n", "Outcome classes:", x$M))
  cat(sprintf("  %-20s%s\n", "Splitting rule:", x$splitting.rule))
  cat(sprintf("  %-20s%s\n", "Sample fraction:", format(x$sample.fraction)))

  if (isTRUE(x$honesty)) {
    cat(sprintf("  %-20s%s\n", "Honesty:",
                paste0("TRUE (fraction = ", x$honesty.fraction, ")")))
  } else {
    cat(sprintf("  %-20s%s\n", "Honesty:", "FALSE"))
  }

  if (!is.null(x$tuning.output)) {
    tp <- x$tuning.output$params
    parts <- vapply(names(tp), function(nm) {
      paste0(nm, " = ", format(tp[[nm]]))
    }, character(1))
    cat(sprintf("  %-20s%s\n", "Tuning:", paste(parts, collapse = ", ")))
  }

  invisible(x)
}


#' Summary method for jocf objects
#'
#' Extends [print.jocf()] with in-sample prediction diagnostics: per-class
#' probability summaries and classification frequency tables.
#'
#' @param object A `"jocf"` object.
#' @param ... Currently unused.
#'
#' @return Invisibly returns an object of class `"summary.jocf"`.
#' @export
summary.jocf <- function(object, ...) {
  M    <- object$M
  preds <- object$predictions

  # Per-class min, mean, max
  prob_summary <- matrix(NA_real_, nrow = 3L, ncol = M)
  for (m in seq_len(M)) {
    prob_summary[1L, m] <- min(preds[, m])
    prob_summary[2L, m] <- mean(preds[, m])
    prob_summary[3L, m] <- max(preds[, m])
  }
  rownames(prob_summary) <- c("Min.", "Mean", "Max.")
  colnames(prob_summary) <- paste0("P(Y=", seq_len(M), ")")

  # Classification tables
  class_prob_tab <- tabulate(object$classification$prob, nbins = M)
  names(class_prob_tab) <- seq_len(M)
  class_vote_tab <- tabulate(object$classification$vote, nbins = M)
  names(class_vote_tab) <- seq_len(M)

  out <- list(
    object         = object,
    prob_summary   = prob_summary,
    class_prob_tab = class_prob_tab,
    class_vote_tab = class_vote_tab
  )
  class(out) <- "summary.jocf"
  out
}


#' Print method for summary.jocf objects
#'
#' @param x A `"summary.jocf"` object.
#' @param digits Number of significant digits. Default `4`.
#' @param ... Currently unused.
#'
#' @return Invisibly returns `x`.
#' @export
print.summary.jocf <- function(x, digits = 4L, ...) {
  print.jocf(x$object)

  cat("\nIn-sample predicted probabilities:\n")
  print(round(x$prob_summary, digits))

  cat("\nIn-sample classification (probability-based):\n")
  print(x$class_prob_tab)

  cat("\nIn-sample classification (majority-vote):\n")
  print(x$class_vote_tab)

  invisible(x)
}
