# test-classification.R
# Tests for probability-based and majority-vote classification.

# Helper: small dataset for fast tests
make_data <- function(n = 100, M = 3, k = 4, seed = 1L) {
  set.seed(seed)
  list(
    Y = sample(seq_len(M), n, replace = TRUE),
    X = matrix(rnorm(n * k), n, k),
    n = n, M = M, k = k
  )
}

# ===========================================================================
# In-sample classification (jocf object)
# ===========================================================================

test_that("jocf: classification$prob is integer vector with values in 1..M", {
  d   <- make_data(n = 100, M = 3, k = 4, seed = 100)
  fit <- jocf(d$Y, d$X, num.trees = 50)
  cp  <- fit$classification$prob
  expect_type(cp, "integer")
  expect_length(cp, d$n)
  expect_true(all(cp >= 1L & cp <= d$M))
})

test_that("jocf: classification$vote is integer vector with values in 1..M", {
  d   <- make_data(n = 100, M = 3, k = 4, seed = 101)
  fit <- jocf(d$Y, d$X, num.trees = 50)
  cv  <- fit$classification$vote
  expect_type(cv, "integer")
  expect_length(cv, d$n)
  expect_true(all(cv >= 1L & cv <= d$M))
})

test_that("jocf: classification$prob matches manual which.max on predictions", {
  d   <- make_data(n = 80, M = 4, k = 3, seed = 102)
  fit <- jocf(d$Y, d$X, num.trees = 60)
  expected <- apply(fit$predictions, 1L, which.max)
  expect_equal(fit$classification$prob, expected)
})

test_that("jocf: classification is a list with $prob and $vote", {
  d   <- make_data(seed = 103)
  fit <- jocf(d$Y, d$X, num.trees = 30)
  expect_type(fit$classification, "list")
  expect_named(fit$classification, c("prob", "vote"))
})

test_that("jocf: classification works with M = 2", {
  d   <- make_data(n = 80, M = 2, k = 3, seed = 104)
  fit <- jocf(d$Y, d$X, num.trees = 50)
  expect_true(all(fit$classification$prob %in% 1:2))
  expect_true(all(fit$classification$vote %in% 1:2))
})

test_that("jocf: classification works with weighted splitting rule", {
  d   <- make_data(n = 100, M = 3, k = 3, seed = 105)
  fit <- jocf(d$Y, d$X, num.trees = 50, splitting.rule = "weighted")
  expect_length(fit$classification$prob, d$n)
  expect_length(fit$classification$vote, d$n)
  expect_true(all(fit$classification$prob >= 1L & fit$classification$prob <= d$M))
  expect_true(all(fit$classification$vote >= 1L & fit$classification$vote <= d$M))
})

# ===========================================================================
# Out-of-sample classification (predict.jocf)
# ===========================================================================

test_that("predict.jocf: returns list with $probabilities and $classification", {
  d     <- make_data(n = 100, M = 3, k = 4, seed = 106)
  fit   <- jocf(d$Y, d$X, num.trees = 50)
  X_new <- matrix(rnorm(20 * 4), 20, 4)
  pred  <- predict(fit, X_new)
  expect_type(pred, "list")
  expect_true("probabilities" %in% names(pred))
  expect_true("classification" %in% names(pred))
  expect_named(pred$classification, c("prob", "vote"))
})

test_that("predict.jocf: classification$prob matches which.max of probabilities", {
  d     <- make_data(n = 100, M = 3, k = 4, seed = 107)
  fit   <- jocf(d$Y, d$X, num.trees = 60)
  X_new <- matrix(rnorm(30 * 4), 30, 4)
  pred  <- predict(fit, X_new)
  expected <- apply(pred$probabilities, 1L, which.max)
  expect_equal(pred$classification$prob, expected)
})

test_that("predict.jocf: classification vectors have correct length", {
  d     <- make_data(n = 80, M = 4, k = 3, seed = 108)
  fit   <- jocf(d$Y, d$X, num.trees = 40)
  X_new <- matrix(rnorm(15 * 3), 15, 3)
  pred  <- predict(fit, X_new)
  expect_length(pred$classification$prob, 15L)
  expect_length(pred$classification$vote, 15L)
})

test_that("predict.jocf: classification values in 1..M", {
  d     <- make_data(n = 100, M = 3, k = 4, seed = 109)
  fit   <- jocf(d$Y, d$X, num.trees = 50)
  X_new <- matrix(rnorm(25 * 4), 25, 4)
  pred  <- predict(fit, X_new)
  expect_true(all(pred$classification$prob >= 1L & pred$classification$prob <= d$M))
  expect_true(all(pred$classification$vote >= 1L & pred$classification$vote <= d$M))
})

# ===========================================================================
# Separable data: both methods should classify correctly
# ===========================================================================

test_that("classification: separable 2-class data yields high accuracy", {
  set.seed(110)
  n <- 300
  X <- matrix(rnorm(n), n, 1)
  Y <- as.integer(ifelse(X[, 1] < 0, 1L, 2L))
  fit <- jocf(Y, X, num.trees = 200, min.node.size = 5)

  acc_prob <- mean(fit$classification$prob == Y)
  acc_vote <- mean(fit$classification$vote == Y)
  expect_gt(acc_prob, 0.8)
  expect_gt(acc_vote, 0.8)
})

test_that("classification: out-of-sample separable data yields high accuracy", {
  set.seed(111)
  n <- 300
  X <- matrix(rnorm(n), n, 1)
  Y <- as.integer(ifelse(X[, 1] < 0, 1L, 2L))
  fit <- jocf(Y, X, num.trees = 200, min.node.size = 5)

  X_new <- matrix(rnorm(100), 100, 1)
  Y_new <- as.integer(ifelse(X_new[, 1] < 0, 1L, 2L))
  pred  <- predict(fit, X_new)

  acc_prob <- mean(pred$classification$prob == Y_new)
  acc_vote <- mean(pred$classification$vote == Y_new)
  expect_gt(acc_prob, 0.8)
  expect_gt(acc_vote, 0.8)
})
