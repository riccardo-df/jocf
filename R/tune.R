#' Validate tuning arguments
#'
#' @param tune.parameters Character: `"none"`, `"all"`, or a character vector of
#'   parameter names.
#' @param tune.num.trees Positive integer; trees per mini-forest.
#' @param tune.num.reps Positive integer; number of mini-forests to evaluate.
#' @param tune.num.draws Positive integer; random candidates evaluated via
#'   Kriging surrogate.
#' @return Invisibly returns the resolved character vector of parameter names
#'   to tune (`character(0)` for `"none"`).
#' @keywords internal
validate_tune_inputs <- function(tune.parameters, tune.num.trees,
                                 tune.num.reps, tune.num.draws) {
  all_tunable <- c("mtry", "min.node.size", "sample.fraction")

  if (length(tune.parameters) == 1L && tune.parameters == "none") {
    return(invisible(character(0)))
  }
  if (length(tune.parameters) == 1L && tune.parameters == "all") {
    tune.parameters <- all_tunable
  }
  if (!is.character(tune.parameters) || length(tune.parameters) == 0L)
    stop('`tune.parameters` must be "none", "all", or a character vector of ',
         "parameter names.", call. = FALSE)
  bad <- setdiff(tune.parameters, all_tunable)
  if (length(bad) > 0L)
    stop(sprintf("Unknown tuning parameter(s): %s. Must be one of: %s.",
                 paste(bad, collapse = ", "),
                 paste(all_tunable, collapse = ", ")), call. = FALSE)

  if (!is.numeric(tune.num.trees) || length(tune.num.trees) != 1L ||
      tune.num.trees < 1L)
    stop("`tune.num.trees` must be a positive integer.", call. = FALSE)
  if (!is.numeric(tune.num.reps) || length(tune.num.reps) != 1L ||
      tune.num.reps < 1L)
    stop("`tune.num.reps` must be a positive integer.", call. = FALSE)
  if (!is.numeric(tune.num.draws) || length(tune.num.draws) != 1L ||
      tune.num.draws < 1L)
    stop("`tune.num.draws` must be a positive integer.", call. = FALSE)

  invisible(tune.parameters)
}


#' Transform unit-interval draws to actual parameter values
#'
#' Each tunable parameter has a dedicated transformation from a uniform draw
#' on the unit interval to the actual parameter space.
#'
#' @param draws Numeric vector of length `d` (one element per tuned parameter,
#'   each in the unit interval).
#' @param param_names Character vector of length `d` giving the names of tuned
#'   parameters (in the same order as `draws`).
#' @param n Integer; number of training observations.
#' @param k Integer; number of covariates.
#' @return Named list of parameter values.
#' @keywords internal
get_params_from_draw <- function(draws, param_names, n, k) {
  params <- list()
  for (idx in seq_along(param_names)) {
    u <- draws[idx]
    nm <- param_names[idx]
    params[[nm]] <- switch(nm,
      mtry = as.integer(max(1L, ceiling(k * u))),
      min.node.size = as.integer(max(1L, floor(2^(u * (log2(n) - 4))))),
      sample.fraction = 0.05 + 0.45 * u
    )
  }
  params
}


#' Built-in hyperparameter tuning engine
#'
#' GRF-style tuning: grow many small mini-forests at random parameter draws,
#' compute debiased OOB error, fit a Kriging surrogate ([DiceKriging::km()]),
#' and select the best parameters.
#'
#' @param Y Integer vector of class labels (1..M).
#' @param X_mat Numeric matrix of covariates (already factor-encoded).
#' @param M Integer; number of outcome classes.
#' @param tune_params Character vector of parameter names to tune.
#' @param defaults Named list of default parameter values
#'   (`mtry`, `min.node.size`, `sample.fraction`).
#' @param tune.num.trees Integer; trees per mini-forest.
#' @param tune.num.reps Integer; number of mini-forests to evaluate.
#' @param tune.num.draws Integer; random candidates evaluated via Kriging.
#' @param max_depth Integer; max tree depth (-1 = unlimited).
#' @param lambda Numeric vector of length M; splitting weights.
#' @param num_threads Integer; OpenMP threads (0 = all).
#' @return Named list with components:
#'   \describe{
#'     \item{`status`}{Character: `"tuned"`, `"default"`, or `"failure"`.}
#'     \item{`params`}{Named list of selected parameter values.}
#'     \item{`error`}{Debiased OOB error at selected parameters.}
#'     \item{`grid`}{Data frame of evaluated draws and errors.}
#'   }
#' @keywords internal
tune_jocf <- function(Y, X_mat, M, tune_params, defaults,
                      tune.num.trees, tune.num.reps, tune.num.draws,
                      max_depth, lambda, num_threads) {

  n <- length(Y)
  k <- ncol(X_mat)
  d <- length(tune_params)

  # Generate tune.num.reps random draws in [0, 1]^d
  draw_matrix <- matrix(stats::runif(tune.num.reps * d), nrow = tune.num.reps,
                        ncol = d)
  colnames(draw_matrix) <- tune_params

  errors <- rep(NA_real_, tune.num.reps)

  for (r in seq_len(tune.num.reps)) {
    params_r <- get_params_from_draw(draw_matrix[r, ], tune_params, n, k)

    # Merge with defaults for non-tuned parameters
    mtry_r     <- if ("mtry" %in% tune_params) params_r$mtry else defaults$mtry
    mns_r      <- if ("min.node.size" %in% tune_params) params_r$min.node.size else defaults$min.node.size
    sf_r       <- if ("sample.fraction" %in% tune_params) params_r$sample.fraction else defaults$sample.fraction
    n_sub_r    <- as.integer(ceiling(sf_r * n))

    res <- tryCatch(
      grow_forest_oob_cpp(
        Y           = Y,
        X           = X_mat,
        num_trees   = as.integer(tune.num.trees),
        min_node_size = as.integer(mns_r),
        max_depth   = max_depth,
        n_sub       = n_sub_r,
        mtry        = as.integer(mtry_r),
        M           = as.integer(M),
        lambda      = lambda,
        num_threads = num_threads
      ),
      error = function(e) NULL
    )
    if (!is.null(res) && is.finite(res$debiased_error)) {
      errors[r] <- res$debiased_error
    }
  }

  # Build grid data frame
  grid <- as.data.frame(draw_matrix)
  grid$error <- errors

  # Filter valid errors
  valid <- which(is.finite(errors))
  if (length(valid) < 10L) {
    warning("Fewer than 10 valid tuning evaluations; using default parameters.",
            call. = FALSE)
    return(list(status = "default", params = defaults,
                error = NA_real_, grid = grid))
  }

  # Check for near-zero variance
  if (stats::sd(errors[valid]) < 1e-15) {
    warning("Near-zero variance in tuning errors; using default parameters.",
            call. = FALSE)
    return(list(status = "default", params = defaults,
                error = NA_real_, grid = grid))
  }

  # Fit Kriging surrogate via DiceKriging
  if (!requireNamespace("DiceKriging", quietly = TRUE)) {
    warning("Package 'DiceKriging' not installed; using default parameters. ",
            "Install it for built-in tuning: install.packages('DiceKriging').",
            call. = FALSE)
    return(list(status = "default", params = defaults,
                error = NA_real_, grid = grid))
  }

  kriging_fit <- tryCatch({
    design <- draw_matrix[valid, , drop = FALSE]
    response <- errors[valid]
    suppressWarnings(
      DiceKriging::km(
        design   = design,
        response = response,
        control  = list(trace = FALSE)
      )
    )
  }, error = function(e) NULL)

  if (is.null(kriging_fit)) {
    # Kriging failed; fall back to best observed draw
    best_idx <- valid[which.min(errors[valid])]
    best_params <- get_params_from_draw(draw_matrix[best_idx, ], tune_params, n, k)
    merged <- defaults
    for (nm in tune_params) merged[[nm]] <- best_params[[nm]]
    return(list(status = "tuned", params = merged,
                error = errors[best_idx], grid = grid))
  }

  # Generate tune.num.draws new random candidates and predict via Kriging
  new_draws <- matrix(stats::runif(tune.num.draws * d), nrow = tune.num.draws,
                      ncol = d)
  colnames(new_draws) <- tune_params

  kriging_pred <- tryCatch({
    DiceKriging::predict.km(kriging_fit, newdata = as.data.frame(new_draws),
                            type = "UK")$mean
  }, error = function(e) NULL)

  if (is.null(kriging_pred) || all(!is.finite(kriging_pred))) {
    # Prediction failed; fall back to best observed draw
    best_idx <- valid[which.min(errors[valid])]
    best_params <- get_params_from_draw(draw_matrix[best_idx, ], tune_params, n, k)
    merged <- defaults
    for (nm in tune_params) merged[[nm]] <- best_params[[nm]]
    return(list(status = "tuned", params = merged,
                error = errors[best_idx], grid = grid))
  }

  # Select draw with lowest predicted error
  best_new <- which.min(kriging_pred)
  best_params <- get_params_from_draw(new_draws[best_new, ], tune_params, n, k)
  merged <- defaults
  for (nm in tune_params) merged[[nm]] <- best_params[[nm]]

  list(status = "tuned", params = merged,
       error = kriging_pred[best_new], grid = grid)
}
