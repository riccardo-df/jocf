# test-honesty.R
# Tests for Phase 4: honesty, variance estimation, and marginal effects SEs.

# ============================================================================
# Helper: small honest forest for reuse
# ============================================================================
make_honest_fit <- function(n = 200, M = 3, B = 50, seed = 42) {
  set.seed(seed)
  X <- matrix(rnorm(n * 3), ncol = 3)
  Y <- sample(seq_len(M), n, replace = TRUE)
  jocf(Y, X, num.trees = B, honesty = TRUE, honesty.fraction = 0.5)
}

# ============================================================================
# Basic honest forest
# ============================================================================
test_that("honest forest: correct dimensions and metadata", {
  fit <- make_honest_fit()
  n <- 200; M <- 3
  expect_s3_class(fit, "jocf")
  expect_true(isTRUE(fit$honesty))
  expect_equal(fit$honesty.fraction, 0.5)
  expect_true(is.list(fit$honest_data))
  expect_equal(fit$honest_data$n_hon, 100L)
  expect_equal(nrow(fit$predictions), n)
  expect_equal(ncol(fit$predictions), M)
})

test_that("honest forest: predictions sum to 1 and are nonneg", {
  fit <- make_honest_fit()
  expect_true(all(fit$predictions >= 0))
  expect_true(all(abs(rowSums(fit$predictions) - 1.0) < 1e-10))
})

test_that("honest forest: classification vectors are correct length and range", {
  fit <- make_honest_fit()
  n <- 200; M <- 3
  expect_length(fit$classification$prob, n)
  expect_length(fit$classification$vote, n)
  expect_true(all(fit$classification$prob >= 1L & fit$classification$prob <= M))
  expect_true(all(fit$classification$vote >= 1L & fit$classification$vote <= M))
})

test_that("honest forest: M = 2 binary case", {
  set.seed(42)
  n <- 150
  X <- matrix(rnorm(n * 2), ncol = 2)
  Y <- sample(1:2, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 30, honesty = TRUE)
  expect_equal(fit$M, 2L)
  expect_equal(ncol(fit$predictions), 2L)
  expect_true(all(abs(rowSums(fit$predictions) - 1.0) < 1e-10))
})

test_that("honest forest: M = 5 case", {
  set.seed(42)
  n <- 300
  X <- matrix(rnorm(n * 4), ncol = 4)
  Y <- sample(1:5, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 30, honesty = TRUE)
  expect_equal(fit$M, 5L)
  expect_equal(ncol(fit$predictions), 5L)
  expect_true(all(abs(rowSums(fit$predictions) - 1.0) < 1e-10))
})

test_that("honest forest: weighted splitting rule", {
  set.seed(42)
  n <- 200
  X <- matrix(rnorm(n * 3), ncol = 3)
  Y <- sample(1:3, n, replace = TRUE, prob = c(0.6, 0.3, 0.1))
  fit <- jocf(Y, X, num.trees = 30, honesty = TRUE, splitting.rule = "weighted")
  expect_equal(fit$splitting.rule, "weighted")
  expect_true(all(fit$predictions >= 0))
  expect_true(all(abs(rowSums(fit$predictions) - 1.0) < 1e-10))
})

test_that("honest forest: max.depth constraint respected", {
  set.seed(42)
  n <- 200
  X <- matrix(rnorm(n * 3), ncol = 3)
  Y <- sample(1:3, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 30, honesty = TRUE, max.depth = 2)
  expect_equal(fit$max.depth, 2)
  expect_true(all(fit$predictions >= 0))
  expect_true(all(abs(rowSums(fit$predictions) - 1.0) < 1e-10))
})

test_that("non-honest forest: honesty field is FALSE, no honest_data", {
  set.seed(42)
  n <- 100
  X <- matrix(rnorm(n * 2), ncol = 2)
  Y <- sample(1:3, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 20, honesty = FALSE)
  expect_false(isTRUE(fit$honesty))
  expect_null(fit$honest_data)
  expect_null(fit$honesty.fraction)
})

# ============================================================================
# Predict with variance
# ============================================================================
test_that("predict with variance: correct dimensions", {
  fit <- make_honest_fit()
  set.seed(99)
  X_new <- matrix(rnorm(20 * 3), ncol = 3)
  pred <- predict(fit, X_new, variance = TRUE)
  expect_true(is.list(pred))
  expect_equal(nrow(pred$probabilities), 20L)
  expect_equal(ncol(pred$probabilities), 3L)
  expect_equal(nrow(pred$variance), 20L)
  expect_equal(ncol(pred$variance), 3L)
  expect_equal(nrow(pred$std.error), 20L)
  expect_equal(ncol(pred$std.error), 3L)
})

test_that("predict with variance: probabilities sum to 1 and nonneg", {
  fit <- make_honest_fit()
  set.seed(99)
  X_new <- matrix(rnorm(20 * 3), ncol = 3)
  pred <- predict(fit, X_new, variance = TRUE)
  expect_true(all(pred$probabilities >= 0))
  expect_true(all(abs(rowSums(pred$probabilities) - 1.0) < 1e-10))
})

test_that("predict with variance: SE nonneg", {
  fit <- make_honest_fit()
  set.seed(99)
  X_new <- matrix(rnorm(20 * 3), ncol = 3)
  pred <- predict(fit, X_new, variance = TRUE)
  expect_true(all(pred$std.error >= 0))
  expect_true(all(pred$variance >= 0))
})

test_that("predict with variance: SE equals sqrt(variance)", {
  fit <- make_honest_fit()
  set.seed(99)
  X_new <- matrix(rnorm(10 * 3), ncol = 3)
  pred <- predict(fit, X_new, variance = TRUE)
  expect_equal(pred$std.error, sqrt(pmax(pred$variance, 0)), tolerance = 1e-14)
})

test_that("predict with variance: classification vectors present", {
  fit <- make_honest_fit()
  set.seed(99)
  X_new <- matrix(rnorm(10 * 3), ncol = 3)
  pred <- predict(fit, X_new, variance = TRUE)
  expect_length(pred$classification$prob, 10)
  expect_length(pred$classification$vote, 10)
  expect_true(all(pred$classification$prob >= 1L & pred$classification$prob <= 3L))
})

test_that("predict variance=TRUE errors on non-honest forest", {
  set.seed(42)
  n <- 100
  X <- matrix(rnorm(n * 2), ncol = 2)
  Y <- sample(1:3, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 20, honesty = FALSE)
  X_new <- matrix(rnorm(5 * 2), ncol = 2)
  expect_error(predict(fit, X_new, variance = TRUE),
               "requires an honest forest")
})

test_that("predict honest without variance: works without variance field", {
  fit <- make_honest_fit()
  set.seed(99)
  X_new <- matrix(rnorm(10 * 3), ncol = 3)
  pred <- predict(fit, X_new, variance = FALSE)
  expect_true(is.list(pred))
  expect_null(pred$variance)
  expect_null(pred$std.error)
  expect_equal(nrow(pred$probabilities), 10L)
  expect_true(all(pred$probabilities >= 0))
  expect_true(all(abs(rowSums(pred$probabilities) - 1.0) < 1e-10))
})

# ============================================================================
# Marginal effects with SEs
# ============================================================================
test_that("marginal_effects honest: AME dimensions and rowSums ~0", {
  fit <- make_honest_fit(n = 200, M = 3, B = 50)
  set.seed(99)
  X <- matrix(rnorm(200 * 3), ncol = 3)
  me <- marginal_effects(fit, X)
  expect_s3_class(me, "jocf_me")
  expect_equal(nrow(me$effects), 3L)
  expect_equal(ncol(me$effects), 3L)
  expect_true(all(abs(rowSums(me$effects)) < 0.1))
})

test_that("marginal_effects honest: SE and CIs present", {
  fit <- make_honest_fit(n = 200, M = 3, B = 50)
  set.seed(99)
  X <- matrix(rnorm(200 * 3), ncol = 3)
  me <- marginal_effects(fit, X)
  expect_false(is.null(me$std.error))
  expect_false(is.null(me$ci.lower))
  expect_false(is.null(me$ci.upper))
  expect_equal(dim(me$std.error), dim(me$effects))
  expect_equal(dim(me$ci.lower), dim(me$effects))
  expect_equal(dim(me$ci.upper), dim(me$effects))
})

test_that("marginal_effects honest: SE nonneg and ci.lower < ci.upper", {
  fit <- make_honest_fit(n = 200, M = 3, B = 50)
  set.seed(99)
  X <- matrix(rnorm(200 * 3), ncol = 3)
  me <- marginal_effects(fit, X)
  expect_true(all(me$std.error >= 0))
  expect_true(all(me$ci.lower <= me$ci.upper))
})

test_that("marginal_effects honest: CIs equal effects +/- 1.96*SE", {
  fit <- make_honest_fit(n = 200, M = 3, B = 50)
  set.seed(99)
  X <- matrix(rnorm(200 * 3), ncol = 3)
  me <- marginal_effects(fit, X)
  expect_equal(me$ci.lower, me$effects - 1.96 * me$std.error, tolerance = 1e-14)
  expect_equal(me$ci.upper, me$effects + 1.96 * me$std.error, tolerance = 1e-14)
})

test_that("marginal_effects honest: eval modes work", {
  fit <- make_honest_fit(n = 200, M = 3, B = 50)
  set.seed(99)
  X <- matrix(rnorm(200 * 3), ncol = 3)
  me_mean <- marginal_effects(fit, X, eval = "mean")
  me_at   <- marginal_effects(fit, X, eval = "atmean")
  me_med  <- marginal_effects(fit, X, eval = "atmedian")
  expect_equal(nrow(me_mean$effects), 3L)
  expect_equal(nrow(me_at$effects), 3L)
  expect_equal(nrow(me_med$effects), 3L)
  expect_false(is.null(me_at$std.error))
  expect_false(is.null(me_med$std.error))
})

test_that("marginal_effects non-honest: no SE or CIs", {
  set.seed(42)
  n <- 100
  X <- matrix(rnorm(n * 2), ncol = 2)
  Y <- sample(1:3, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 20, honesty = FALSE)
  me <- marginal_effects(fit, X)
  expect_null(me$std.error)
  expect_null(me$ci.lower)
  expect_null(me$ci.upper)
})

# ============================================================================
# Validation
# ============================================================================
test_that("honesty.fraction must be in (0, 1)", {
  set.seed(42)
  n <- 100
  X <- matrix(rnorm(n * 2), ncol = 2)
  Y <- sample(1:3, n, replace = TRUE)
  expect_error(jocf(Y, X, num.trees = 10, honesty = TRUE, honesty.fraction = 0),
               "honesty.fraction")
  expect_error(jocf(Y, X, num.trees = 10, honesty = TRUE, honesty.fraction = 1),
               "honesty.fraction")
  expect_error(jocf(Y, X, num.trees = 10, honesty = TRUE, honesty.fraction = -0.5),
               "honesty.fraction")
  expect_error(jocf(Y, X, num.trees = 10, honesty = TRUE, honesty.fraction = "a"),
               "honesty.fraction")
})

test_that("honesty must be logical", {
  set.seed(42)
  n <- 100
  X <- matrix(rnorm(n * 2), ncol = 2)
  Y <- sample(1:3, n, replace = TRUE)
  expect_error(jocf(Y, X, num.trees = 10, honesty = "yes"),
               "honesty")
})

# ============================================================================
# Signal tests
# ============================================================================
test_that("honest forest: separable data still yields good accuracy", {
  set.seed(42)
  n <- 300
  X <- matrix(rnorm(n * 2), ncol = 2)
  # Easy classification: Y depends on X[,1]
  Y <- ifelse(X[, 1] < -0.5, 1L, ifelse(X[, 1] < 0.5, 2L, 3L))
  fit <- jocf(Y, X, num.trees = 100, honesty = TRUE, honesty.fraction = 0.5)
  # In-sample accuracy (probability-based) should be decent
  acc <- mean(fit$classification$prob == Y)
  expect_gt(acc, 0.4)  # well above 1/3 chance
})

test_that("SEs decrease with more trees", {
  set.seed(42)
  n <- 200
  X <- matrix(rnorm(n * 2), ncol = 2)
  Y <- sample(1:3, n, replace = TRUE)
  X_new <- matrix(rnorm(10 * 2), ncol = 2)

  fit_small <- jocf(Y, X, num.trees = 50, honesty = TRUE)
  fit_large <- jocf(Y, X, num.trees = 200, honesty = TRUE)
  pred_small <- predict(fit_small, X_new, variance = TRUE)
  pred_large <- predict(fit_large, X_new, variance = TRUE)

  # Average SE should be smaller with more trees
  mean_se_small <- mean(pred_small$std.error)
  mean_se_large <- mean(pred_large$std.error)
  expect_lt(mean_se_large, mean_se_small)
})

test_that("honest vs non-honest predictions are correlated", {
  set.seed(42)
  n <- 200
  X <- matrix(rnorm(n * 2), ncol = 2)
  Y <- sample(1:3, n, replace = TRUE)
  X_new <- matrix(rnorm(50 * 2), ncol = 2)

  fit_nh <- jocf(Y, X, num.trees = 100, honesty = FALSE)
  fit_h  <- jocf(Y, X, num.trees = 100, honesty = TRUE)
  pred_nh <- predict(fit_nh, X_new)
  pred_h  <- predict(fit_h, X_new)

  # Predictions for class 1 should be positively correlated
  corr <- cor(pred_nh$probabilities[, 1], pred_h$probabilities[, 1])
  expect_gt(corr, 0.3)
})

# ============================================================================
# Edge cases
# ============================================================================
test_that("extreme honesty.fraction: 0.1 (small honesty sample)", {
  set.seed(42)
  n <- 200
  X <- matrix(rnorm(n * 2), ncol = 2)
  Y <- sample(1:3, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 30, honesty = TRUE, honesty.fraction = 0.1)
  expect_equal(fit$honest_data$n_hon, 20L)
  expect_true(all(fit$predictions >= 0))
  expect_true(all(abs(rowSums(fit$predictions) - 1.0) < 1e-10))
})

test_that("extreme honesty.fraction: 0.9 (large honesty sample)", {
  set.seed(42)
  n <- 200
  X <- matrix(rnorm(n * 2), ncol = 2)
  Y <- sample(1:3, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 30, honesty = TRUE, honesty.fraction = 0.9)
  expect_equal(fit$honest_data$n_hon, 180L)
  expect_true(all(fit$predictions >= 0))
  expect_true(all(abs(rowSums(fit$predictions) - 1.0) < 1e-10))
})

test_that("single test observation", {
  fit <- make_honest_fit()
  X_new <- matrix(rnorm(3), nrow = 1)
  pred <- predict(fit, X_new, variance = TRUE)
  expect_equal(nrow(pred$probabilities), 1L)
  expect_equal(nrow(pred$variance), 1L)
  expect_true(all(pred$std.error >= 0))
})

test_that("honest forest with factor covariates", {
  set.seed(42)
  n <- 200
  X <- data.frame(x1 = rnorm(n), x2 = factor(sample(c("A", "B", "C"), n, TRUE)))
  Y <- sample(1:3, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 30, honesty = TRUE)
  expect_true(isTRUE(fit$honesty))
  expect_true(all(fit$predictions >= 0))
  expect_true(all(abs(rowSums(fit$predictions) - 1.0) < 1e-10))

  # Predict with factors
  X_new <- data.frame(x1 = rnorm(5), x2 = factor(sample(c("A", "B", "C"), 5, TRUE)))
  pred <- predict(fit, X_new, variance = TRUE)
  expect_equal(nrow(pred$probabilities), 5L)
  expect_true(all(pred$std.error >= 0))
})

test_that("print.jocf_me displays SEs when present", {
  fit <- make_honest_fit(n = 200, M = 3, B = 30)
  set.seed(99)
  X <- matrix(rnorm(200 * 3), ncol = 3)
  me <- marginal_effects(fit, X)
  output <- capture.output(print(me))
  expect_true(any(grepl("Standard Errors", output)))
})

test_that("print.jocf_me does NOT display SEs for non-honest", {
  set.seed(42)
  n <- 100
  X <- matrix(rnorm(n * 2), ncol = 2)
  Y <- sample(1:3, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 20, honesty = FALSE)
  me <- marginal_effects(fit, X)
  output <- capture.output(print(me))
  expect_false(any(grepl("Standard Errors", output)))
})

# ============================================================================
# summary.jocf_me (honest)
# ============================================================================

test_that("summary.jocf_me honest: shows per-class tables with z values", {
  fit <- make_honest_fit(n = 200, M = 3, B = 30, seed = 200)
  set.seed(201)
  X <- matrix(rnorm(200 * 3), ncol = 3)
  me <- marginal_effects(fit, X)
  out <- capture.output(summary(me))
  expect_true(any(grepl("--- P\\(Y=1\\) ---", out)))
  expect_true(any(grepl("--- P\\(Y=2\\) ---", out)))
  expect_true(any(grepl("--- P\\(Y=3\\) ---", out)))
  expect_true(any(grepl("z value", out)))
  expect_true(any(grepl("Std.Error", out)))
  expect_true(any(grepl("CI 95%", out)))
})

test_that("summary.jocf_me honest: returns invisibly", {
  fit <- make_honest_fit(n = 200, M = 3, B = 30, seed = 202)
  set.seed(203)
  X <- matrix(rnorm(200 * 3), ncol = 3)
  me <- marginal_effects(fit, X)
  out <- capture.output(res <- summary(me))
  expect_identical(res, me)
})
