# ===========================================================================
# Tests for hyperparameter tuning
# ===========================================================================

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------
set.seed(42)
n_test <- 200
X_test <- matrix(rnorm(n_test * 4), ncol = 4,
                 dimnames = list(NULL, paste0("x", 1:4)))
latent <- X_test[, 1] - 0.5 * X_test[, 2] + rnorm(n_test)
Y_test <- as.integer(cut(latent, breaks = c(-Inf, -0.5, 0.5, Inf)))

# ===========================================================================
# Parameter transforms
# ===========================================================================

test_that("get_params_from_draw: u=0 produces valid params", {
  p <- get_params_from_draw(c(0, 0, 0),
                            c("mtry", "min.node.size", "sample.fraction"),
                            n = 200, k = 4)
  # ceiling(4 * 0) = 0, then max(1L, 0) = 1
  expect_equal(p$mtry, 1L)
  expect_true(is.integer(p$mtry))
  expect_true(is.integer(p$min.node.size))
  expect_gte(p$min.node.size, 1L)
  expect_equal(p$sample.fraction, 0.05)
})

test_that("get_params_from_draw: u=1 produces valid params", {
  p <- get_params_from_draw(c(1, 1, 1),
                            c("mtry", "min.node.size", "sample.fraction"),
                            n = 200, k = 4)
  expect_equal(p$mtry, 4L)
  expect_true(is.integer(p$min.node.size))
  expect_gte(p$min.node.size, 1L)
  expect_equal(p$sample.fraction, 0.50)
})

test_that("get_params_from_draw: mtry is always integer in {1, ..., k}", {
  for (u in c(0.01, 0.25, 0.5, 0.75, 0.99)) {
    p <- get_params_from_draw(u, "mtry", n = 200, k = 10)
    expect_true(is.integer(p$mtry))
    expect_gte(p$mtry, 1L)
    expect_lte(p$mtry, 10L)
  }
})

test_that("get_params_from_draw: sample.fraction in [0.05, 0.50]", {
  for (u in seq(0, 1, by = 0.1)) {
    p <- get_params_from_draw(u, "sample.fraction", n = 200, k = 4)
    expect_gte(p$sample.fraction, 0.05)
    expect_lte(p$sample.fraction, 0.50)
  }
})

test_that("get_params_from_draw: min.node.size >= 1", {
  for (u in seq(0, 1, by = 0.1)) {
    p <- get_params_from_draw(u, "min.node.size", n = 200, k = 4)
    expect_true(is.integer(p$min.node.size))
    expect_gte(p$min.node.size, 1L)
  }
})

# ===========================================================================
# Input validation
# ===========================================================================

test_that("validate_tune_inputs: 'none' returns empty character", {
  result <- validate_tune_inputs("none", 50, 100, 1000)
  expect_identical(result, character(0))
})

test_that("validate_tune_inputs: 'all' returns all three params", {
  result <- validate_tune_inputs("all", 50, 100, 1000)
  expect_identical(result, c("mtry", "min.node.size", "sample.fraction"))
})

test_that("validate_tune_inputs: subset works", {
  result <- validate_tune_inputs(c("mtry", "sample.fraction"), 50, 100, 1000)
  expect_identical(result, c("mtry", "sample.fraction"))
})

test_that("validate_tune_inputs: invalid param name errors", {
  expect_error(validate_tune_inputs("bogus", 50, 100, 1000),
               "Unknown tuning parameter")
})

test_that("validate_tune_inputs: tune.num.trees < 1 errors", {
  expect_error(validate_tune_inputs("all", 0, 100, 1000),
               "tune.num.trees")
})

test_that("validate_tune_inputs: tune.num.reps < 1 errors", {
  expect_error(validate_tune_inputs("all", 50, 0, 1000),
               "tune.num.reps")
})

test_that("validate_tune_inputs: tune.num.draws < 1 errors", {
  expect_error(validate_tune_inputs("all", 50, 100, 0),
               "tune.num.draws")
})

# ===========================================================================
# OOB predictions (grow_forest_oob_cpp)
# ===========================================================================

test_that("grow_forest_oob_cpp: returns correct structure", {
  M <- max(Y_test)
  lambda <- rep(1.0, M)
  n_sub <- as.integer(ceiling(0.5 * n_test))
  res <- grow_forest_oob_cpp(Y_test, X_test, num_trees = 20L,
                              min_node_size = 5L, alpha = 0.05, max_depth = -1L,
                              n_sub = n_sub, mtry = 2L, M = M,
                              lambda = lambda, num_threads = 1L)
  expect_true(is.list(res))
  expect_true("oob_predictions" %in% names(res))
  expect_true("debiased_error" %in% names(res))
})

test_that("grow_forest_oob_cpp: OOB predictions have correct dimensions", {
  M <- max(Y_test)
  lambda <- rep(1.0, M)
  n_sub <- as.integer(ceiling(0.5 * n_test))
  res <- grow_forest_oob_cpp(Y_test, X_test, num_trees = 30L,
                              min_node_size = 5L, alpha = 0.05, max_depth = -1L,
                              n_sub = n_sub, mtry = 2L, M = M,
                              lambda = lambda, num_threads = 1L)
  expect_equal(nrow(res$oob_predictions), n_test)
  expect_equal(ncol(res$oob_predictions), M)
})

test_that("grow_forest_oob_cpp: valid OOB predictions are non-negative and sum to ~1", {
  M <- max(Y_test)
  lambda <- rep(1.0, M)
  n_sub <- as.integer(ceiling(0.5 * n_test))
  res <- grow_forest_oob_cpp(Y_test, X_test, num_trees = 50L,
                              min_node_size = 5L, alpha = 0.05, max_depth = -1L,
                              n_sub = n_sub, mtry = 2L, M = M,
                              lambda = lambda, num_threads = 1L)
  valid_rows <- which(!is.nan(res$oob_predictions[, 1]))
  expect_true(length(valid_rows) > 0)
  for (i in valid_rows) {
    row <- res$oob_predictions[i, ]
    expect_true(all(row >= -1e-10))
    expect_equal(sum(row), 1.0, tolerance = 1e-10)
  }
})

test_that("grow_forest_oob_cpp: debiased error is finite", {
  M <- max(Y_test)
  lambda <- rep(1.0, M)
  n_sub <- as.integer(ceiling(0.5 * n_test))
  res <- grow_forest_oob_cpp(Y_test, X_test, num_trees = 50L,
                              min_node_size = 5L, alpha = 0.05, max_depth = -1L,
                              n_sub = n_sub, mtry = 2L, M = M,
                              lambda = lambda, num_threads = 1L)
  expect_true(is.finite(res$debiased_error))
})

test_that("grow_forest_oob_cpp: most observations have OOB predictions", {
  M <- max(Y_test)
  lambda <- rep(1.0, M)
  n_sub <- as.integer(ceiling(0.5 * n_test))
  res <- grow_forest_oob_cpp(Y_test, X_test, num_trees = 50L,
                              min_node_size = 5L, alpha = 0.05, max_depth = -1L,
                              n_sub = n_sub, mtry = 2L, M = M,
                              lambda = lambda, num_threads = 1L)
  valid_rows <- which(!is.nan(res$oob_predictions[, 1]))
  # With 50 trees and 50% subsampling, virtually all obs should be OOB at least twice
  expect_gt(length(valid_rows), n_test * 0.9)
})

# ===========================================================================
# End-to-end tuning (requires DiceKriging)
# ===========================================================================

test_that("jocf with tune.parameters='none' has NULL tuning.output", {
  fit <- jocf(Y_test, X_test, num.trees = 20, tune.parameters = "none")
  expect_null(fit$tuning.output)
})

skip_if_not_installed("DiceKriging")

test_that("jocf with tune.parameters='all' runs successfully", {
  fit <- jocf(Y_test, X_test, num.trees = 50,
              tune.parameters = "all",
              tune.num.trees = 10, tune.num.reps = 20,
              tune.num.draws = 50)
  expect_s3_class(fit, "jocf")
  expect_false(is.null(fit$tuning.output))
})

test_that("tuning.output has expected structure", {
  fit <- jocf(Y_test, X_test, num.trees = 50,
              tune.parameters = "all",
              tune.num.trees = 10, tune.num.reps = 20,
              tune.num.draws = 50)
  to <- fit$tuning.output
  expect_true(to$status %in% c("tuned", "default", "failure"))
  expect_true(is.list(to$params))
  expect_true("mtry" %in% names(to$params))
  expect_true("min.node.size" %in% names(to$params))
  expect_true("sample.fraction" %in% names(to$params))
  expect_true(is.data.frame(to$grid))
  expect_true("error" %in% names(to$grid))
})

test_that("tuned params are in valid ranges", {
  k <- ncol(X_test)
  fit <- jocf(Y_test, X_test, num.trees = 50,
              tune.parameters = "all",
              tune.num.trees = 10, tune.num.reps = 20,
              tune.num.draws = 50)
  p <- fit$tuning.output$params
  expect_gte(p$mtry, 1L)
  expect_lte(p$mtry, k)
  expect_gte(p$min.node.size, 1L)
  expect_gte(p$sample.fraction, 0.05)
  expect_lte(p$sample.fraction, 0.50)
})

test_that("tuning a subset of parameters works", {
  fit <- jocf(Y_test, X_test, num.trees = 50,
              tune.parameters = c("mtry"),
              tune.num.trees = 10, tune.num.reps = 20,
              tune.num.draws = 50)
  expect_false(is.null(fit$tuning.output))
  # Grid should have only one draw column plus error
  expect_equal(ncol(fit$tuning.output$grid), 2L)
})

test_that("tuning with small data: falls back gracefully", {
  set.seed(99)
  n_small <- 20
  X_small <- matrix(rnorm(n_small * 2), ncol = 2)
  Y_small <- sample(1:2, n_small, replace = TRUE)
  # With very few reps some may be invalid → should still return
  fit <- suppressWarnings(
    jocf(Y_small, X_small, num.trees = 20,
         tune.parameters = "all",
         tune.num.trees = 5, tune.num.reps = 15,
         tune.num.draws = 20)
  )
  expect_s3_class(fit, "jocf")
})

# ===========================================================================
# Integration: tuned forest works with predict and marginal_effects
# ===========================================================================

test_that("predict() works on tuned forest", {
  fit <- jocf(Y_test, X_test, num.trees = 50,
              tune.parameters = "all",
              tune.num.trees = 10, tune.num.reps = 20,
              tune.num.draws = 50)
  preds <- predict(fit, X_test[1:10, ])
  expect_equal(nrow(preds$probabilities), 10)
  expect_equal(ncol(preds$probabilities), fit$M)
  expect_true(all(preds$probabilities >= 0))
  expect_equal(rowSums(preds$probabilities), rep(1, 10), tolerance = 1e-10)
})

test_that("marginal_effects() works on tuned forest", {
  fit <- jocf(Y_test, X_test, num.trees = 50,
              tune.parameters = "all",
              tune.num.trees = 10, tune.num.reps = 20,
              tune.num.draws = 50)
  me <- marginal_effects(fit, X_test)
  expect_true(all(is.finite(me$effects)))
})

test_that("in-sample predictions from tuned forest are valid", {
  fit <- jocf(Y_test, X_test, num.trees = 50,
              tune.parameters = "all",
              tune.num.trees = 10, tune.num.reps = 20,
              tune.num.draws = 50)
  expect_true(all(fit$predictions >= 0))
  expect_equal(rowSums(fit$predictions), rep(1, n_test), tolerance = 1e-10)
  expect_equal(ncol(fit$predictions), fit$M)
})
