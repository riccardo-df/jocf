# test-evaluation.R
# Tests for evaluation metrics: probability + classification.

# ===========================================================================
# Probability metrics — mean_squared_error
# ===========================================================================

test_that("MSE: perfect indicator predictions give 0", {
  y <- c(1L, 2L, 3L, 1L, 3L)
  P <- matrix(0, 5, 3)
  P[cbind(1:5, y)] <- 1
  expect_equal(mean_squared_error(y, P), 0)
})

test_that("MSE: uniform predictions on balanced data have known value", {
  # For each obs with M=3: rowSum((indicator - 1/3)^2) = (2/3)^2 + 2*(1/3)^2 = 6/9 = 2/3
  y <- c(1L, 2L, 3L)
  P <- matrix(1/3, 3, 3)
  expect_equal(mean_squared_error(y, P), 2/3)
})

test_that("MSE: use.true = TRUE with perfect match gives 0", {
  true_probs <- matrix(c(0.7, 0.2, 0.1, 0.3, 0.3, 0.4), 2, 3, byrow = TRUE)
  expect_equal(mean_squared_error(true_probs, true_probs, use.true = TRUE), 0)
})

test_that("MSE: use.true = TRUE with known difference", {
  # Two obs, M=2. true = [0.8, 0.2; 0.6, 0.4], pred = [0.5, 0.5; 0.5, 0.5]
  # obs1: (0.3)^2 + (-0.3)^2 = 0.18
  # obs2: (0.1)^2 + (-0.1)^2 = 0.02
  # mean = 0.10
  true_p <- matrix(c(0.8, 0.2, 0.6, 0.4), 2, 2, byrow = TRUE)
  pred   <- matrix(0.5, 2, 2)
  expect_equal(mean_squared_error(true_p, pred, use.true = TRUE), 0.10)
})

# ===========================================================================
# Probability metrics — mean_absolute_error
# ===========================================================================

test_that("MAE: perfect indicator predictions give 0", {
  y <- c(1L, 2L, 3L)
  P <- matrix(0, 3, 3)
  P[cbind(1:3, y)] <- 1
  expect_equal(mean_absolute_error(y, P), 0)
})

test_that("MAE: uniform predictions on balanced data have known value", {
  # For each obs with M=3: rowSum(|indicator - 1/3|) = 2/3 + 1/3 + 1/3 = 4/3
  y <- c(1L, 2L, 3L)
  P <- matrix(1/3, 3, 3)
  expect_equal(mean_absolute_error(y, P), 4/3)
})

test_that("MAE: use.true = TRUE works correctly", {
  true_p <- matrix(c(0.8, 0.2, 0.6, 0.4), 2, 2, byrow = TRUE)
  pred   <- matrix(0.5, 2, 2)
  # obs1: |0.3| + |-0.3| = 0.6; obs2: |0.1| + |-0.1| = 0.2; mean = 0.4
  expect_equal(mean_absolute_error(true_p, pred, use.true = TRUE), 0.4)
})

# ===========================================================================
# Probability metrics — mean_ranked_score
# ===========================================================================

test_that("RPS: perfect indicator predictions give 0", {
  y <- c(1L, 2L, 3L)
  P <- matrix(0, 3, 3)
  P[cbind(1:3, y)] <- 1
  expect_equal(mean_ranked_score(y, P), 0)
})

test_that("RPS: known hand-computed value for M=3", {
  # y = 1, pred = (0, 0, 1) — worst possible for ordered metric
  # CDF_true = (1, 1), CDF_pred = (0, 0)
  # sum of squared diffs = 1 + 1 = 2; RPS = 2/(3-1) = 1
  y <- 1L
  P <- matrix(c(0, 0, 1), 1, 3)
  expect_equal(mean_ranked_score(y, P), 1)
})

test_that("RPS: penalises ordering-violating predictions more than MSE", {
  # y = 1; pred_close = (0, 1, 0); pred_far = (0, 0, 1)
  # Both have same MSE (= 2) but RPS differs
  y <- 1L
  P_close <- matrix(c(0, 1, 0), 1, 3)
  P_far   <- matrix(c(0, 0, 1), 1, 3)
  expect_equal(mean_squared_error(y, P_close), mean_squared_error(y, P_far))
  expect_lt(mean_ranked_score(y, P_close), mean_ranked_score(y, P_far))
})

test_that("RPS: use.true = TRUE with perfect match gives 0", {
  true_p <- matrix(c(0.5, 0.3, 0.2, 0.1, 0.4, 0.5), 2, 3, byrow = TRUE)
  expect_equal(mean_ranked_score(true_p, true_p, use.true = TRUE), 0)
})

# ===========================================================================
# Classification metrics — classification_error
# ===========================================================================

test_that("CE: perfect classification gives 0", {
  y <- c(1L, 2L, 3L, 1L, 2L)
  expect_equal(classification_error(y, y), 0)
})

test_that("CE: all wrong gives 1", {
  y    <- c(1L, 1L, 1L)
  yhat <- c(2L, 2L, 2L)
  expect_equal(classification_error(y, yhat), 1)
})

test_that("CE: known fraction", {
  y    <- c(1L, 2L, 3L, 1L, 2L)
  yhat <- c(1L, 2L, 2L, 1L, 3L)
  expect_equal(classification_error(y, yhat), 2/5)
})

# ===========================================================================
# Classification metrics — mean_absolute_class_error
# ===========================================================================

test_that("MACE: perfect classification gives 0", {
  y <- c(1L, 2L, 3L)
  expect_equal(mean_absolute_class_error(y, y), 0)
})

test_that("MACE: penalises distant misclassifications more", {
  y     <- c(1L, 1L)
  close <- c(2L, 2L)  # |1-2| = 1 each → mean = 1
  far   <- c(3L, 3L)  # |1-3| = 2 each → mean = 2
  expect_lt(mean_absolute_class_error(y, close),
            mean_absolute_class_error(y, far))
})

test_that("MACE: known value", {
  y    <- c(1L, 2L, 5L)
  yhat <- c(3L, 2L, 1L)
  # |1-3| + |2-2| + |5-1| = 2 + 0 + 4 = 6; mean = 2
  expect_equal(mean_absolute_class_error(y, yhat), 2)
})

# ===========================================================================
# Classification metrics — weighted_kappa
# ===========================================================================

test_that("weighted_kappa: perfect agreement gives 1", {
  y <- c(1L, 2L, 3L, 1L, 2L)
  expect_equal(weighted_kappa(y, y, M = 3), 1)
})

test_that("weighted_kappa: quadratic vs linear differ", {
  y    <- c(1L, 2L, 3L, 1L, 2L, 3L)
  yhat <- c(1L, 1L, 2L, 2L, 3L, 3L)
  kq <- weighted_kappa(y, yhat, M = 3, type = "quadratic")
  kl <- weighted_kappa(y, yhat, M = 3, type = "linear")
  expect_false(isTRUE(all.equal(kq, kl)))
})

test_that("weighted_kappa: known quadratic value for 2x2 case", {
  # M=2, y = (1,1,2,2), yhat = (1,2,1,2)
  # Confusion matrix O (normalised): [[0.25, 0.25], [0.25, 0.25]]
  # W quadratic: [[0, 1], [1, 0]]
  # sum(W*O) = 0.5
  # hist_y = (0.5, 0.5), hist_p = (0.5, 0.5)
  # E = [[0.25, 0.25], [0.25, 0.25]]
  # sum(W*E) = 0.5
  # kappa = 1 - 0.5/0.5 = 0
  y    <- c(1L, 1L, 2L, 2L)
  yhat <- c(1L, 2L, 1L, 2L)
  expect_equal(weighted_kappa(y, yhat, M = 2), 0)
})

test_that("weighted_kappa: returns NA when denominator is zero", {
  # All same class → both histograms are (1, 0, 0), E is all zeros except [1,1]
  # W[1,1] = 0, so sum(W*E) = 0 → NA
  y    <- c(1L, 1L, 1L)
  yhat <- c(1L, 1L, 1L)
  expect_true(is.na(weighted_kappa(y, yhat, M = 3)))
})

test_that("weighted_kappa: M parameter handles unobserved classes", {
  y    <- c(1L, 1L, 2L, 2L)
  yhat <- c(1L, 2L, 1L, 2L)
  # Should work with M=5 even though only classes 1-2 observed
  k5 <- weighted_kappa(y, yhat, M = 5)
  k2 <- weighted_kappa(y, yhat, M = 2)
  expect_type(k5, "double")
  expect_type(k2, "double")
})

# ===========================================================================
# Input validation
# ===========================================================================

test_that("MSE: rejects mismatched dimensions", {
  y <- c(1L, 2L)
  P <- matrix(1/3, 3, 3)
  expect_error(mean_squared_error(y, P), "same number")
})

test_that("MSE: rejects y values outside 1..M", {
  y <- c(0L, 1L, 2L)
  P <- matrix(1/3, 3, 3)
  expect_error(mean_squared_error(y, P), "1, ..., M")
})

test_that("MAE: rejects matrix y when use.true = FALSE", {
  y <- matrix(1/3, 3, 3)
  P <- matrix(1/3, 3, 3)
  expect_error(mean_absolute_error(y, P, use.true = FALSE), "integer vector")
})

test_that("CE: rejects mismatched lengths", {
  expect_error(classification_error(1:3, 1:2), "same length")
})

test_that("MACE: rejects mismatched lengths", {
  expect_error(mean_absolute_class_error(1:3, 1:2), "same length")
})

test_that("weighted_kappa: rejects invalid type", {
  expect_error(weighted_kappa(1:3, 1:3, M = 3, type = "cubic"), "quadratic")
})

test_that("weighted_kappa: rejects M < 2", {
  expect_error(weighted_kappa(1L, 1L, M = 1), "M.*>= 2")
})
