# test-predictions.R
# Phase 3 tests: marginal_effects.jocf() — new interface with eval / target_covariates

# Helper: fits a small jocf and returns training data too
make_fit <- function(n = 120, M = 3, k = 4, num.trees = 80, seed = 1L) {
  set.seed(seed)
  Y <- sample(seq_len(M), n, replace = TRUE)
  X <- matrix(rnorm(n * k), n, k)
  fit <- jocf(Y, X, num.trees = num.trees, min.node.size = 5)
  list(fit = fit, X = X, Y = Y, n = n, M = M, k = k)
}

# ===========================================================================
# eval = "mean" (default)
# ===========================================================================

test_that("marginal_effects eval=mean: AME has correct dimensions (k x M)", {
  d  <- make_fit(seed = 1)
  me <- marginal_effects(d$fit, data = d$X, eval = "mean")
  expect_equal(dim(me$effects), c(d$k, d$M))
})

test_that("marginal_effects eval=mean: returns a jocf_me object", {
  d  <- make_fit(seed = 2)
  me <- marginal_effects(d$fit, data = d$X, eval = "mean")
  expect_s3_class(me, "jocf_me")
})

test_that("marginal_effects eval=mean: all AME values are finite", {
  d  <- make_fit(seed = 3)
  me <- marginal_effects(d$fit, data = d$X, eval = "mean")
  expect_true(all(is.finite(me$effects)))
})

test_that("marginal_effects eval=mean: AME rows sum to 0 across classes", {
  d  <- make_fit(n = 100, M = 4, k = 3, seed = 4)
  me <- marginal_effects(d$fit, data = d$X, eval = "mean")
  expect_equal(unname(rowSums(me$effects)), rep(0, d$k), tolerance = 1e-10)
})

test_that("marginal_effects eval=mean: eval field stored correctly", {
  d  <- make_fit(seed = 5)
  me <- marginal_effects(d$fit, data = d$X, eval = "mean")
  expect_identical(me$eval, "mean")
})

# ===========================================================================
# eval = "atmean"
# ===========================================================================

test_that("marginal_effects eval=atmean: AME has correct dimensions (k x M)", {
  d  <- make_fit(seed = 6)
  me <- marginal_effects(d$fit, data = d$X, eval = "atmean")
  expect_equal(dim(me$effects), c(d$k, d$M))
})

test_that("marginal_effects eval=atmean: all AME values are finite", {
  d  <- make_fit(seed = 7)
  me <- marginal_effects(d$fit, data = d$X, eval = "atmean")
  expect_true(all(is.finite(me$effects)))
})

test_that("marginal_effects eval=atmean: AME rows sum to 0 across classes", {
  d  <- make_fit(n = 100, M = 3, k = 4, seed = 8)
  me <- marginal_effects(d$fit, data = d$X, eval = "atmean")
  expect_equal(unname(rowSums(me$effects)), rep(0, d$k), tolerance = 1e-10)
})

# ===========================================================================
# eval = "atmedian"
# ===========================================================================

test_that("marginal_effects eval=atmedian: AME has correct dimensions (k x M)", {
  d  <- make_fit(seed = 9)
  me <- marginal_effects(d$fit, data = d$X, eval = "atmedian")
  expect_equal(dim(me$effects), c(d$k, d$M))
})

test_that("marginal_effects eval=atmedian: all AME values are finite", {
  d  <- make_fit(seed = 10)
  me <- marginal_effects(d$fit, data = d$X, eval = "atmedian")
  expect_true(all(is.finite(me$effects)))
})

test_that("marginal_effects eval=atmedian: AME rows sum to 0 across classes", {
  d  <- make_fit(n = 100, M = 3, k = 4, seed = 11)
  me <- marginal_effects(d$fit, data = d$X, eval = "atmedian")
  expect_equal(unname(rowSums(me$effects)), rep(0, d$k), tolerance = 1e-10)
})

# ===========================================================================
# target_covariates subset
# ===========================================================================

test_that("marginal_effects target_covariates=c(1,3): AME has 2 rows", {
  d  <- make_fit(n = 120, M = 3, k = 4, seed = 12)
  me <- marginal_effects(d$fit, data = d$X, target_covariates = c(1L, 3L))
  expect_equal(dim(me$effects), c(2L, d$M))
})

test_that("marginal_effects target_covariates subset: AME rows sum to 0", {
  d  <- make_fit(n = 120, M = 3, k = 4, seed = 13)
  me <- marginal_effects(d$fit, data = d$X, target_covariates = c(2L, 4L))
  expect_equal(unname(rowSums(me$effects)), c(0, 0), tolerance = 1e-10)
})

test_that("marginal_effects target_covariates: target_covariates field stored", {
  d  <- make_fit(seed = 14)
  me <- marginal_effects(d$fit, data = d$X, target_covariates = c(1L, 3L))
  expect_equal(me$target_covariates, c(1L, 3L))
})

test_that("marginal_effects target_covariates: error on out-of-range index", {
  d <- make_fit(n = 80, M = 3, k = 4, seed = 15)
  expect_error(
    marginal_effects(d$fit, data = d$X, target_covariates = 5L),
    regexp = "1..ncol"
  )
})

# ===========================================================================
# Discrete covariates
# ===========================================================================

test_that("marginal_effects discrete_vars: runs without error", {
  d      <- make_fit(n = 100, M = 3, k = 4, seed = 16)
  X2     <- d$X
  X2[,2] <- sample(0L:3L, d$n, replace = TRUE)
  expect_no_error(
    marginal_effects(d$fit, data = X2, discrete_vars = 2L)
  )
})

test_that("marginal_effects discrete_vars: AME rows sum to 0", {
  set.seed(17)
  n <- 120; M <- 3; k <- 3
  Y <- sample(seq_len(M), n, replace = TRUE)
  X <- matrix(rnorm(n * k), n, k)
  X[, 3] <- sample(0L:2L, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 60)
  me  <- marginal_effects(fit, data = X, discrete_vars = 3L)
  expect_equal(unname(rowSums(me$effects)), rep(0, k), tolerance = 1e-10)
  expect_true(all(is.finite(me$effects)))
})

# ===========================================================================
# Input validation
# ===========================================================================

test_that("marginal_effects: error on wrong number of data columns", {
  d    <- make_fit(n = 80, M = 3, k = 4, seed = 18)
  Xbad <- matrix(rnorm(d$n * 3), d$n, 3)
  expect_error(marginal_effects(d$fit, data = Xbad), regexp = "4 column")
})

test_that("marginal_effects: error on non-positive bandwidth", {
  d <- make_fit(seed = 19)
  expect_error(marginal_effects(d$fit, data = d$X, bandwidth = -0.1),
               regexp = "bandwidth")
})

test_that("marginal_effects: error on invalid eval argument", {
  d <- make_fit(seed = 20)
  expect_error(marginal_effects(d$fit, data = d$X, eval = "bad"),
               regexp = "arg")
})

# ===========================================================================
# print method
# ===========================================================================

test_that("print.jocf_me: runs without error and returns invisibly", {
  d   <- make_fit(n = 80, M = 3, k = 2, seed = 21)
  me  <- marginal_effects(d$fit, data = d$X)
  out <- capture.output(res <- print(me))
  expect_identical(res, me)
  expect_true(any(grepl("Average Marginal", out)))
})

test_that("print.jocf_me: eval label appears in output", {
  d   <- make_fit(seed = 22)
  me  <- marginal_effects(d$fit, data = d$X, eval = "atmean")
  out <- capture.output(print(me))
  expect_true(any(grepl("Marginal Effects at Mean", out)))
})
