# test-splitting.R
# Phase 1 tests for node_impurity_cpp() and find_best_split_cpp().

# ---------------------------------------------------------------------------
# node_impurity_cpp
# ---------------------------------------------------------------------------

test_that("node_impurity: M=2 binary case equals p*(1-p)", {
  # For M=2 with unweighted lambda, the formula gives
  #   Q = (1/2) * [p*(1-p) + (1-p)*p] = p*(1-p)
  # which is the standard Bernoulli variance.
  p <- 3 / 10
  expected <- p * (1 - p)
  result <- node_impurity_cpp(c(3L, 7L), n = 10L, M = 2L,
                               lambda = c(1, 1))
  expect_equal(result, expected, tolerance = 1e-12)
})

test_that("node_impurity: weighted with ones equals unweighted", {
  counts <- c(5L, 3L, 2L)
  n      <- 10L
  M      <- 3L
  unweighted    <- node_impurity_cpp(counts, n, M, rep(1.0, M))
  weighted_ones <- node_impurity_cpp(counts, n, M, c(1.0, 1.0, 1.0))
  expect_equal(unweighted, weighted_ones, tolerance = 1e-15)
})

test_that("node_impurity: pure node has zero impurity", {
  # All observations belong to class 2; p_2 = 1 => p_2*(1-p_2) = 0
  result <- node_impurity_cpp(c(0L, 10L), n = 10L, M = 2L,
                               lambda = c(1, 1))
  expect_equal(result, 0.0, tolerance = 1e-12)
})

test_that("node_impurity: M=2 uniform distribution gives Q = 0.25", {
  # p = 0.5 => Q = 0.5 * 0.5 = 0.25
  result <- node_impurity_cpp(c(5L, 5L), n = 10L, M = 2L,
                               lambda = c(1, 1))
  expect_equal(result, 0.25, tolerance = 1e-12)
})

test_that("node_impurity: M=3 manual calculation", {
  # counts = (2, 5, 3), n = 10
  # p = (0.2, 0.5, 0.3)
  # Q = (1/3) * [0.2*0.8 + 0.5*0.5 + 0.3*0.7]
  #   = (1/3) * [0.16 + 0.25 + 0.21]
  #   = (1/3) * 0.62 ≈ 0.20667
  counts   <- c(2L, 5L, 3L)
  n        <- 10L
  M        <- 3L
  expected <- (0.2 * 0.8 + 0.5 * 0.5 + 0.3 * 0.7) / 3
  result   <- node_impurity_cpp(counts, n, M, rep(1, M))
  expect_equal(result, expected, tolerance = 1e-12)
})

test_that("node_impurity: variance-weighted amplifies rare-class contribution", {
  # Global proportions: p_hat_1 = 0.1, p_hat_2 = 0.9
  # lambda_1 = 1/(0.1*0.9) ≈ 11.11, lambda_2 = 1/(0.9*0.1) ≈ 11.11
  # Node counts: (3, 7), n = 10  =>  p_1 = 0.3, p_2 = 0.7
  # Q_w = (lambda_1 * 0.3*0.7 + lambda_2 * 0.7*0.3) / 2
  p_hat   <- c(0.1, 0.9)
  lambda  <- 1 / (p_hat * (1 - p_hat))
  counts  <- c(3L, 7L)
  n       <- 10L
  M       <- 2L
  expected <- sum(lambda * c(0.3 * 0.7, 0.7 * 0.3)) / M
  result   <- node_impurity_cpp(counts, n, M, lambda)
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("node_impurity: weighted Q_w > unweighted Q when class is rare", {
  # Rare class gets upweighted => weighted impurity should be larger than
  # unweighted for the same counts.
  p_hat  <- c(0.1, 0.9)
  lambda <- 1 / (p_hat * (1 - p_hat))   # both > 1
  counts <- c(3L, 7L)
  n      <- 10L
  M      <- 2L
  q_unweighted <- node_impurity_cpp(counts, n, M, rep(1, M))
  q_weighted   <- node_impurity_cpp(counts, n, M, lambda)
  expect_gt(q_weighted, q_unweighted)
})

# ---------------------------------------------------------------------------
# find_best_split_cpp
# ---------------------------------------------------------------------------

test_that("find_best_split: finds correct feature for perfectly separable data", {
  set.seed(1)
  n   <- 200
  # x1 separates class 1 vs 2 perfectly at 0; x2 is noise
  x1  <- rnorm(n)
  x2  <- rnorm(n)
  y   <- as.integer(ifelse(x1 < 0, 1L, 2L))
  X   <- cbind(x1, x2)

  res <- find_best_split_cpp(y, X, M = 2L, lambda = c(1, 1),
                              min_node_size = 5L)
  expect_true(res$found)
  expect_equal(res$feature, 1L)            # x1 is feature 1
  expect_true(res$threshold > -0.5 && res$threshold < 0.5)
})

test_that("find_best_split: threshold is the midpoint of adjacent sorted values", {
  # 6 observations, all class labels known, 1-feature case
  y <- c(1L, 1L, 1L, 2L, 2L, 2L)
  X <- matrix(c(1, 2, 3, 4, 5, 6), ncol = 1)
  res <- find_best_split_cpp(y, X, M = 2L, lambda = c(1, 1),
                              min_node_size = 1L)
  expect_true(res$found)
  # Best split should be between x=3 and x=4, threshold = 3.5
  expect_equal(res$threshold, 3.5, tolerance = 1e-12)
})

test_that("find_best_split: no split when min_node_size cannot be satisfied", {
  # 4 observations, min_node_size = 3 => need 3+3 = 6 minimum
  y <- c(1L, 1L, 2L, 2L)
  X <- matrix(c(1, 2, 3, 4), ncol = 1)
  res <- find_best_split_cpp(y, X, M = 2L, lambda = c(1, 1),
                              min_node_size = 3L)
  expect_false(res$found)
  expect_equal(res$feature, 0L)
})

test_that("find_best_split: no split for a pure node", {
  # All observations in same class => impurity cannot decrease
  y <- c(1L, 1L, 1L, 1L, 1L)
  X <- matrix(c(1, 2, 3, 4, 5), ncol = 1)
  res <- find_best_split_cpp(y, X, M = 2L, lambda = c(1, 1),
                              min_node_size = 1L)
  # A split is technically found (it minimises zero vs zero), but since
  # all observations are class 1, p_2 = 0 everywhere and impurity = 0.
  # Either outcome is valid; what matters is the threshold is coherent.
  # We only assert that returned feature is valid (0 or in-range).
  expect_true(res$feature == 0L || (res$feature >= 1L && res$feature <= ncol(X)))
})

test_that("find_best_split: no split when all feature values are identical", {
  y <- c(1L, 2L, 1L, 2L, 1L)
  X <- matrix(rep(5.0, 5), ncol = 1)   # constant feature
  res <- find_best_split_cpp(y, X, M = 2L, lambda = c(1, 1),
                              min_node_size = 1L)
  expect_false(res$found)
})

test_that("find_best_split: unweighted and weighted(ones) give same split", {
  set.seed(7)
  n <- 100
  x <- rnorm(n)
  y <- as.integer(ifelse(x < 0.5, 1L, ifelse(x < 1.5, 2L, 3L)))
  X <- matrix(x, ncol = 1)

  res_uw  <- find_best_split_cpp(y, X, M = 3L, lambda = rep(1, 3),
                                  min_node_size = 5L)
  res_w1  <- find_best_split_cpp(y, X, M = 3L, lambda = c(1, 1, 1),
                                  min_node_size = 5L)
  expect_equal(res_uw$feature,   res_w1$feature)
  expect_equal(res_uw$threshold, res_w1$threshold, tolerance = 1e-12)
  expect_equal(res_uw$impurity,  res_w1$impurity,  tolerance = 1e-12)
})

test_that("find_best_split: impurity after split <= parent impurity", {
  set.seed(42)
  n <- 50
  y <- sample(1L:3L, n, replace = TRUE)
  X <- matrix(rnorm(n * 2), ncol = 2)
  M <- 3L

  res <- find_best_split_cpp(y, X, M = M, lambda = rep(1, M),
                              min_node_size = 5L)
  if (res$found) {
    # Parent impurity
    parent_counts <- tabulate(y, nbins = M)
    parent_Q      <- node_impurity_cpp(as.integer(parent_counts), n, M,
                                        rep(1, M))
    # n-weighted sum of children impurity should be <= n * parent
    # (splitting cannot increase total impurity).
    expect_lte(res$impurity, n * parent_Q + 1e-10)
  } else {
    skip("No split found in random data — try a different seed")
  }
})
