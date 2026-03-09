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
  expect_equal(dim(pred$probabilities), c(20L, d$M))
})

test_that("predict.jocf: out-of-sample row sums equal 1", {
  d     <- make_data(n = 100, M = 3, k = 4, seed = 12)
  fit   <- jocf(d$Y, d$X, num.trees = 50)
  X_new <- matrix(rnorm(30 * 4), 30, 4)
  pred  <- predict(fit, newdata = X_new)
  expect_equal(rowSums(pred$probabilities), rep(1, 30), tolerance = 1e-10)
})

test_that("predict.jocf: out-of-sample predictions are non-negative", {
  d     <- make_data(n = 100, M = 2, k = 3, seed = 13)
  fit   <- jocf(d$Y, d$X, num.trees = 50)
  X_new <- matrix(rnorm(50 * 3), 50, 3)
  pred  <- predict(fit, newdata = X_new)
  expect_true(all(pred$probabilities >= 0))
})

test_that("predict.jocf: error on wrong number of columns", {
  d   <- make_data(n = 80, M = 3, k = 4, seed = 14)
  fit <- jocf(d$Y, d$X, num.trees = 20)
  X_bad <- matrix(rnorm(10 * 3), 10, 3)   # 3 cols instead of 4
  expect_error(predict(fit, newdata = X_bad), regexp = "4 column")
})

# ---------------------------------------------------------------------------
# max.depth
# ---------------------------------------------------------------------------

test_that("jocf: max.depth = 1 produces stumps (at most 3 nodes per tree)", {
  d   <- make_data(n = 200, M = 3, k = 4, seed = 30)
  fit <- jocf(d$Y, d$X, num.trees = 50, max.depth = 1)
  for (b in seq_along(fit$forest)) {
    tree <- fit$forest[[b]]
    expect_lte(length(tree$split_feature), 3L)
  }
  # Predictions still valid
  expect_equal(rowSums(fit$predictions), rep(1, d$n), tolerance = 1e-10)
  expect_true(all(fit$predictions >= 0))
})

test_that("jocf: max.depth = 2 produces at most 7 nodes per tree", {
  d   <- make_data(n = 200, M = 3, k = 4, seed = 31)
  fit <- jocf(d$Y, d$X, num.trees = 50, max.depth = 2)
  for (b in seq_along(fit$forest)) {
    tree <- fit$forest[[b]]
    expect_lte(length(tree$split_feature), 7L)
  }
  expect_equal(rowSums(fit$predictions), rep(1, d$n), tolerance = 1e-10)
  expect_true(all(fit$predictions >= 0))
})

test_that("jocf: max.depth = NULL (default) grows deeper trees than max.depth = 1", {
  d    <- make_data(n = 200, M = 3, k = 4, seed = 32)
  fit1 <- jocf(d$Y, d$X, num.trees = 50, max.depth = 1)
  fit0 <- jocf(d$Y, d$X, num.trees = 50)
  # Default should have more nodes on average
  avg_nodes_1 <- mean(vapply(fit1$forest, function(t) length(t$split_feature), 1L))
  avg_nodes_0 <- mean(vapply(fit0$forest, function(t) length(t$split_feature), 1L))
  expect_gt(avg_nodes_0, avg_nodes_1)
})

test_that("jocf: max.depth stored in returned object", {
  d   <- make_data(n = 60, M = 2, k = 3, seed = 33)
  fit <- jocf(d$Y, d$X, num.trees = 10, max.depth = 3)
  expect_equal(fit$max.depth, 3L)
  fit2 <- jocf(d$Y, d$X, num.trees = 10)
  expect_null(fit2$max.depth)
})

test_that("jocf: max.depth validation rejects invalid values", {
  d <- make_data(n = 60, M = 2, k = 3, seed = 34)
  expect_error(jocf(d$Y, d$X, num.trees = 10, max.depth = 0),
               "positive integer")
  expect_error(jocf(d$Y, d$X, num.trees = 10, max.depth = -1),
               "positive integer")
  expect_error(jocf(d$Y, d$X, num.trees = 10, max.depth = 1.5),
               "positive integer")
  expect_error(jocf(d$Y, d$X, num.trees = 10, max.depth = "abc"),
               "positive integer")
})

# ---------------------------------------------------------------------------
# sample.fraction
# ---------------------------------------------------------------------------

test_that("jocf: sample.fraction = 0.5 produces valid predictions", {
  d   <- make_data(n = 200, M = 3, k = 4, seed = 40)
  fit <- jocf(d$Y, d$X, num.trees = 50, sample.fraction = 0.5)
  expect_equal(dim(fit$predictions), c(d$n, d$M))
  expect_equal(rowSums(fit$predictions), rep(1, d$n), tolerance = 1e-10)
  expect_true(all(fit$predictions >= 0))
})

test_that("jocf: sample.fraction = 1.0 works (full sample, no replacement)", {
  d   <- make_data(n = 100, M = 3, k = 4, seed = 41)
  fit <- jocf(d$Y, d$X, num.trees = 30, sample.fraction = 1.0)
  expect_equal(dim(fit$predictions), c(d$n, d$M))
  expect_equal(rowSums(fit$predictions), rep(1, d$n), tolerance = 1e-10)
  expect_true(all(fit$predictions >= 0))
})

test_that("jocf: sample.fraction is stored in returned object", {
  d   <- make_data(n = 60, M = 2, k = 3, seed = 42)
  fit <- jocf(d$Y, d$X, num.trees = 10, sample.fraction = 0.7)
  expect_equal(fit$sample.fraction, 0.7)
})

test_that("jocf: sample.fraction validation rejects invalid values", {
  d <- make_data(n = 60, M = 2, k = 3, seed = 43)
  expect_error(jocf(d$Y, d$X, num.trees = 10, sample.fraction = 0),
               "\\(0, 1\\]")
  expect_error(jocf(d$Y, d$X, num.trees = 10, sample.fraction = -0.5),
               "\\(0, 1\\]")
  expect_error(jocf(d$Y, d$X, num.trees = 10, sample.fraction = 1.5),
               "\\(0, 1\\]")
  expect_error(jocf(d$Y, d$X, num.trees = 10, sample.fraction = "half"),
               "\\(0, 1\\]")
})

# ---------------------------------------------------------------------------
# print.jocf
# ---------------------------------------------------------------------------

test_that("print.jocf: runs without error and returns invisibly", {
  d   <- make_data(n = 80, M = 3, k = 4, seed = 50)
  fit <- jocf(d$Y, d$X, num.trees = 20)
  out <- capture.output(res <- print(fit))
  expect_identical(res, fit)
  expect_true(any(grepl("Joint Ordered Correlation Forest", out)))
})

test_that("print.jocf: displays key metadata", {
  d   <- make_data(n = 100, M = 4, k = 5, seed = 51)
  fit <- jocf(d$Y, d$X, num.trees = 30, splitting.rule = "weighted")
  out <- capture.output(print(fit))
  expect_true(any(grepl("Number of trees:.*30", out)))
  expect_true(any(grepl("Observations:.*100", out)))
  expect_true(any(grepl("Covariates:.*5", out)))
  expect_true(any(grepl("Outcome classes:.*4", out)))
  expect_true(any(grepl("Splitting rule:.*weighted", out)))
  expect_true(any(grepl("Honesty:.*FALSE", out)))
})

test_that("print.jocf: shows sample fraction", {
  d   <- make_data(n = 80, M = 3, k = 3, seed = 52)
  fit <- jocf(d$Y, d$X, num.trees = 10, sample.fraction = 0.7)
  out <- capture.output(print(fit))
  expect_true(any(grepl("Sample fraction:.*0.7", out)))
})

test_that("print.jocf: shows honesty when TRUE", {
  d   <- make_data(n = 200, M = 3, k = 3, seed = 53)
  fit <- jocf(d$Y, d$X, num.trees = 20, honesty = TRUE, honesty.fraction = 0.4)
  out <- capture.output(print(fit))
  expect_true(any(grepl("Honesty:.*TRUE.*fraction.*0.4", out)))
})

test_that("print.jocf: omits Tuning line when no tuning", {
  d   <- make_data(n = 80, M = 3, k = 3, seed = 54)
  fit <- jocf(d$Y, d$X, num.trees = 10)
  out <- capture.output(print(fit))
  expect_false(any(grepl("Tuning:", out)))
})

# ---------------------------------------------------------------------------
# summary.jocf
# ---------------------------------------------------------------------------

test_that("summary.jocf: returns a summary.jocf object", {
  d   <- make_data(n = 100, M = 3, k = 4, seed = 55)
  fit <- jocf(d$Y, d$X, num.trees = 20)
  s   <- summary(fit)
  expect_s3_class(s, "summary.jocf")
})

test_that("summary.jocf: prob_summary has correct dimensions", {
  d   <- make_data(n = 100, M = 4, k = 3, seed = 56)
  fit <- jocf(d$Y, d$X, num.trees = 20)
  s   <- summary(fit)
  expect_equal(dim(s$prob_summary), c(3L, 4L))
  expect_equal(rownames(s$prob_summary), c("Min.", "Mean", "Max."))
})

test_that("summary.jocf: classification tables have correct structure", {
  d   <- make_data(n = 100, M = 3, k = 4, seed = 57)
  fit <- jocf(d$Y, d$X, num.trees = 20)
  s   <- summary(fit)
  expect_equal(length(s$class_prob_tab), 3L)
  expect_equal(length(s$class_vote_tab), 3L)
  expect_equal(sum(s$class_prob_tab), 100L)
  expect_equal(sum(s$class_vote_tab), 100L)
})

test_that("print.summary.jocf: displays prediction diagnostics", {
  d   <- make_data(n = 80, M = 3, k = 3, seed = 58)
  fit <- jocf(d$Y, d$X, num.trees = 20)
  s   <- summary(fit)
  out <- capture.output(print(s))
  expect_true(any(grepl("In-sample predicted probabilities", out)))
  expect_true(any(grepl("probability-based", out)))
  expect_true(any(grepl("majority-vote", out)))
  expect_true(any(grepl("Joint Ordered Correlation Forest", out)))
})

test_that("summary.jocf: Min <= Mean <= Max for each class", {
  d   <- make_data(n = 150, M = 3, k = 4, seed = 59)
  fit <- jocf(d$Y, d$X, num.trees = 50)
  s   <- summary(fit)
  for (m in seq_len(3L)) {
    expect_lte(s$prob_summary["Min.", m], s$prob_summary["Mean", m])
    expect_lte(s$prob_summary["Mean", m], s$prob_summary["Max.", m])
  }
})
