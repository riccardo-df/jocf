# test-forest.R
# Phase 2 tests: jocf() wrapper, in-sample predictions, predict.jocf().

# Helper: small dataset for fast tests
make_data <- function(n = 100, M = 3, k = 4, seed = 1L) {
  set.seed(seed)
  list(
    Y = sample(seq_len(M), n, replace = TRUE),
    X = matrix(rnorm(n * k), n, k),
    n = n, M = M, k = k
  )
}

# ---------------------------------------------------------------------------
# Dimensions and basic structure
# ---------------------------------------------------------------------------

test_that("jocf: predictions have correct dimensions", {
  d   <- make_data(n = 100, M = 3, k = 4, seed = 1)
  fit <- jocf(d$Y, d$X, num.trees = 50, min.node.size = 5)
  expect_equal(dim(fit$predictions), c(d$n, d$M))
})

test_that("jocf: returns an object of class 'jocf'", {
  d   <- make_data(seed = 2)
  fit <- jocf(d$Y, d$X, num.trees = 20)
  expect_s3_class(fit, "jocf")
})

test_that("jocf: stores M and num.trees correctly", {
  d   <- make_data(n = 80, M = 4, seed = 3)
  fit <- jocf(d$Y, d$X, num.trees = 17L)
  expect_equal(fit$M,         4L)
  expect_equal(fit$num.trees, 17L)
})

test_that("jocf: stores k (number of training columns)", {
  d   <- make_data(n = 60, M = 2, k = 7, seed = 4)
  fit <- jocf(d$Y, d$X, num.trees = 10)
  expect_equal(fit$k, 7L)
})

# ---------------------------------------------------------------------------
# Probability validity: rowSums == 1 and non-negativity
# ---------------------------------------------------------------------------

test_that("jocf: in-sample row sums equal 1 (M = 2)", {
  d   <- make_data(n = 200, M = 2, k = 3, seed = 5)
  fit <- jocf(d$Y, d$X, num.trees = 100)
  expect_equal(rowSums(fit$predictions), rep(1, d$n), tolerance = 1e-10)
})

test_that("jocf: in-sample row sums equal 1 (M = 4)", {
  d   <- make_data(n = 150, M = 4, k = 5, seed = 6)
  fit <- jocf(d$Y, d$X, num.trees = 80)
  expect_equal(rowSums(fit$predictions), rep(1, d$n), tolerance = 1e-10)
})

test_that("jocf: all in-sample predictions are non-negative", {
  d   <- make_data(n = 120, M = 3, k = 4, seed = 7)
  fit <- jocf(d$Y, d$X, num.trees = 50, min.node.size = 3)
  expect_true(all(fit$predictions >= 0))
})

# ---------------------------------------------------------------------------
# Splitting rules
# ---------------------------------------------------------------------------

test_that("jocf: weighted splitting rule runs without error", {
  d <- make_data(n = 100, M = 3, k = 3, seed = 8)
  expect_no_error(jocf(d$Y, d$X, num.trees = 20, splitting.rule = "weighted"))
})

test_that("jocf: weighted predictions also sum to 1 and are non-negative", {
  d   <- make_data(n = 100, M = 3, k = 3, seed = 9)
  fit <- jocf(d$Y, d$X, num.trees = 50, splitting.rule = "weighted")
  expect_equal(rowSums(fit$predictions), rep(1, d$n), tolerance = 1e-10)
  expect_true(all(fit$predictions >= 0))
})

# ---------------------------------------------------------------------------
# Predictive signal: separable data
# ---------------------------------------------------------------------------

test_that("jocf: identifies correct class for perfectly separable 2-class data", {
  set.seed(10)
  n <- 300
  X <- matrix(rnorm(n), n, 1)
  Y <- as.integer(ifelse(X[, 1] < 0, 1L, 2L))
  fit <- jocf(Y, X, num.trees = 200, min.node.size = 5)
  # Mean predicted probability for the true class should be well above 0.7
  correct_p <- vapply(seq_len(n), function(i) fit$predictions[i, Y[i]], 1.0)
  expect_gt(mean(correct_p), 0.7)
})

# ---------------------------------------------------------------------------
# predict.jocf()
# ---------------------------------------------------------------------------

test_that("predict.jocf: out-of-sample dimensions are correct", {
  d     <- make_data(n = 100, M = 3, k = 4, seed = 11)
  fit   <- jocf(d$Y, d$X, num.trees = 50)
  X_new <- matrix(rnorm(20 * 4), 20, 4)
  pred  <- predict(fit, newdata = X_new)
  expect_equal(dim(pred), c(20L, d$M))
})

test_that("predict.jocf: out-of-sample row sums equal 1", {
  d     <- make_data(n = 100, M = 3, k = 4, seed = 12)
  fit   <- jocf(d$Y, d$X, num.trees = 50)
  X_new <- matrix(rnorm(30 * 4), 30, 4)
  pred  <- predict(fit, newdata = X_new)
  expect_equal(rowSums(pred), rep(1, 30), tolerance = 1e-10)
})

test_that("predict.jocf: out-of-sample predictions are non-negative", {
  d     <- make_data(n = 100, M = 2, k = 3, seed = 13)
  fit   <- jocf(d$Y, d$X, num.trees = 50)
  X_new <- matrix(rnorm(50 * 3), 50, 3)
  pred  <- predict(fit, newdata = X_new)
  expect_true(all(pred >= 0))
})

test_that("predict.jocf: error on wrong number of columns", {
  d   <- make_data(n = 80, M = 3, k = 4, seed = 14)
  fit <- jocf(d$Y, d$X, num.trees = 20)
  X_bad <- matrix(rnorm(10 * 3), 10, 3)   # 3 cols instead of 4
  expect_error(predict(fit, newdata = X_bad), regexp = "4 column")
})
