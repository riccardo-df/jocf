# ---- encode_factors() unit tests ---------------------------------------------

test_that("numeric matrix passes through unchanged", {
  X <- matrix(rnorm(30), ncol = 3)
  res <- encode_factors(X, Y = rep(1:3, 10))
  expect_identical(res$X_encoded, X)
  expect_null(res$factor_info)
})

test_that("numeric data.frame passes through (returns matrix)", {
  X <- data.frame(a = rnorm(10), b = rnorm(10))
  res <- encode_factors(X, Y = rep(1:2, 5))
  expect_true(is.matrix(res$X_encoded))
  expect_true(is.numeric(res$X_encoded))
  expect_null(res$factor_info)
})

test_that("unordered factor levels are sorted by mean(Y)", {
  Y <- c(1L, 1L, 3L, 3L, 2L, 2L)
  X <- data.frame(f = factor(c("A", "A", "B", "B", "C", "C")))
  res <- encode_factors(X, Y = Y)

  # mean(Y) by level: A=1, C=2, B=3 → sorted order: A, C, B
  expect_equal(res$factor_info[[1]]$levels, c("A", "C", "B"))
  # A→1, C→2, B→3
  expect_equal(res$X_encoded[, 1], c(1, 1, 3, 3, 2, 2))
})

test_that("ordered factor preserves level order", {
  X <- data.frame(f = ordered(c("low", "med", "high", "low"),
                              levels = c("low", "med", "high")))
  res <- encode_factors(X, Y = c(1L, 2L, 3L, 1L))

  expect_equal(res$factor_info[[1]]$type, "ordered")
  expect_equal(res$factor_info[[1]]$levels, c("low", "med", "high"))
  expect_equal(res$X_encoded[, 1], c(1, 2, 3, 1))
})

test_that("logical columns become 1/2", {
  X <- data.frame(flag = c(FALSE, TRUE, FALSE, TRUE))
  res <- encode_factors(X, Y = c(1L, 2L, 1L, 2L))

  expect_equal(res$factor_info[[1]]$type, "logical")
  expect_equal(res$X_encoded[, 1], c(1, 2, 1, 2))
})

test_that("character column gives an error", {
  X <- data.frame(s = c("a", "b"), stringsAsFactors = FALSE)
  expect_error(encode_factors(X, Y = 1:2), "character.*Convert to factor")
})

test_that("mixed data.frame encodes correctly", {
  X <- data.frame(
    num = c(1.5, 2.5, 3.5, 4.5),
    fac = factor(c("A", "B", "A", "B")),
    lgl = c(TRUE, FALSE, TRUE, FALSE)
  )
  Y <- c(1L, 2L, 1L, 2L)
  res <- encode_factors(X, Y = Y)

  expect_true(is.numeric(res$X_encoded))
  expect_equal(ncol(res$X_encoded), 3)
  # Numeric column unchanged
  expect_equal(res$X_encoded[, 1], c(1.5, 2.5, 3.5, 4.5))
  # factor_info: NULL for numeric, non-NULL for factor and logical
  expect_null(res$factor_info[[1]])
  expect_equal(res$factor_info[[2]]$type, "unordered")
  expect_equal(res$factor_info[[3]]$type, "logical")
})

# ---- Prediction-mode re-encoding --------------------------------------------

test_that("prediction mode re-encodes factor levels correctly", {
  X_train <- data.frame(f = factor(c("A", "B", "C", "A", "B", "C")))
  Y <- c(1L, 2L, 3L, 1L, 2L, 3L)
  res_train <- encode_factors(X_train, Y = Y)

  X_new <- data.frame(f = factor(c("C", "A", "B")))
  res_pred <- encode_factors(X_new, factor_info = res_train$factor_info)

  # Codes should match training ordering
  code_C <- match("C", res_train$factor_info[[1]]$levels)
  code_A <- match("A", res_train$factor_info[[1]]$levels)
  code_B <- match("B", res_train$factor_info[[1]]$levels)
  expect_equal(res_pred$X_encoded[, 1], c(code_C, code_A, code_B))
})

test_that("unseen factor level produces warning and median code", {
  X_train <- data.frame(f = factor(c("A", "B", "C", "A")))
  Y <- c(1L, 2L, 3L, 1L)
  res_train <- encode_factors(X_train, Y = Y)

  X_new <- data.frame(f = factor(c("A", "D"), levels = c("A", "D")))
  expect_warning(
    res_pred <- encode_factors(X_new, factor_info = res_train$factor_info),
    "unseen level"
  )
  # D gets median code = ceiling(3/2) = 2
  expect_equal(unname(res_pred$X_encoded[2, 1]), 2)
})

test_that("type mismatch at prediction gives an error", {
  X_train <- data.frame(f = factor(c("A", "B")))
  res_train <- encode_factors(X_train, Y = 1:2)

  X_new <- data.frame(f = c(1.0, 2.0))
  expect_error(encode_factors(X_new, factor_info = res_train$factor_info),
               "training had factor.*got numeric")
})

test_that("prediction with NULL factor_info coerces to matrix", {
  X_new <- data.frame(a = c(1.0, 2.0), b = c(3.0, 4.0))
  res <- encode_factors(X_new, factor_info = NULL)
  expect_true(is.matrix(res$X_encoded))
  expect_true(is.numeric(res$X_encoded))
})

# ---- jocf() with factors ----------------------------------------------------

test_that("jocf() runs with factor covariates", {
  set.seed(42)
  n <- 100
  X <- data.frame(
    x1 = rnorm(n),
    x2 = factor(sample(c("A", "B", "C"), n, replace = TRUE)),
    x3 = sample(c(TRUE, FALSE), n, replace = TRUE)
  )
  Y <- sample(1:3, n, replace = TRUE)

  fit <- jocf(Y, X, num.trees = 20)
  expect_s3_class(fit, "jocf")
  expect_equal(nrow(fit$predictions), n)
  expect_equal(ncol(fit$predictions), 3)
  # Rows sum to 1
  expect_true(all(abs(rowSums(fit$predictions) - 1) < 1e-10))
  # Nonneg
  expect_true(all(fit$predictions >= 0))
})

test_that("jocf() stores factor_info", {
  set.seed(1)
  n <- 60
  X <- data.frame(
    x1 = rnorm(n),
    x2 = factor(sample(c("A", "B"), n, replace = TRUE))
  )
  Y <- sample(1:2, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 10)

  expect_false(is.null(fit$factor_info))
  expect_null(fit$factor_info[[1]])        # numeric column
  expect_equal(fit$factor_info[[2]]$type, "unordered")
})

test_that("jocf() with pure numeric matrix has NULL factor_info", {
  set.seed(1)
  n <- 50
  X <- matrix(rnorm(n * 2), ncol = 2)
  Y <- sample(1:2, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 10)
  expect_null(fit$factor_info)
})

test_that("jocf() with all-factor data.frame works", {
  set.seed(1)
  n <- 80
  X <- data.frame(
    f1 = factor(sample(c("a", "b", "c"), n, replace = TRUE)),
    f2 = ordered(sample(c("lo", "hi"), n, replace = TRUE), levels = c("lo", "hi"))
  )
  Y <- sample(1:3, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 10)

  expect_equal(ncol(fit$predictions), 3)
  expect_true(all(abs(rowSums(fit$predictions) - 1) < 1e-10))
})

# ---- predict.jocf() with factors --------------------------------------------

test_that("predict.jocf() works with factor newdata", {
  set.seed(42)
  n <- 80
  X <- data.frame(
    x1 = rnorm(n),
    x2 = factor(sample(c("A", "B"), n, replace = TRUE))
  )
  Y <- sample(1:3, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 20)

  X_new <- data.frame(
    x1 = rnorm(10),
    x2 = factor(sample(c("A", "B"), 10, replace = TRUE))
  )
  preds <- predict(fit, X_new)
  expect_equal(nrow(preds$probabilities), 10)
  expect_equal(ncol(preds$probabilities), 3)
  expect_true(all(abs(rowSums(preds$probabilities) - 1) < 1e-10))
  expect_true(all(preds$probabilities >= 0))
})

test_that("predict.jocf() warns on unseen levels", {
  set.seed(1)
  n <- 60
  X <- data.frame(x = factor(sample(c("A", "B", "C"), n, replace = TRUE)))
  Y <- sample(1:2, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 10)

  X_new <- data.frame(x = factor(c("A", "Z"), levels = c("A", "Z")))
  expect_warning(predict(fit, X_new), "unseen level")
})

test_that("predict.jocf() errors on type mismatch", {
  set.seed(1)
  n <- 60
  X <- data.frame(x = factor(sample(c("A", "B"), n, replace = TRUE)))
  Y <- sample(1:2, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 10)

  X_new <- data.frame(x = c(1.0, 2.0))
  expect_error(predict(fit, X_new), "training had factor")
})

# ---- marginal_effects() with factors ----------------------------------------

test_that("marginal_effects() auto-excludes factor columns with message", {
  set.seed(42)
  n <- 100
  X <- data.frame(
    x1 = rnorm(n),
    x2 = factor(sample(c("A", "B"), n, replace = TRUE)),
    x3 = rnorm(n)
  )
  Y <- sample(1:3, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 20)

  expect_message(
    me <- marginal_effects(fit, X),
    "excluded from marginal effects"
  )
  # Only numeric covariates in AME
  expect_equal(nrow(me$effects), 2)
  expect_equal(rownames(me$effects), c("x1", "x3"))
})

test_that("marginal_effects() warns when user requests factor column", {
  set.seed(42)
  n <- 80
  X <- data.frame(
    x1 = rnorm(n),
    x2 = factor(sample(c("A", "B"), n, replace = TRUE))
  )
  Y <- sample(1:2, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 10)

  expect_warning(
    me <- marginal_effects(fit, X, target_covariates = c(1, 2)),
    "excluded from marginal effects"
  )
  expect_equal(nrow(me$effects), 1)
  expect_equal(rownames(me$effects), "x1")
})

test_that("marginal_effects() errors when all covariates are factors", {
  set.seed(1)
  n <- 60
  X <- data.frame(
    f1 = factor(sample(c("A", "B"), n, replace = TRUE)),
    f2 = factor(sample(c("X", "Y"), n, replace = TRUE))
  )
  Y <- sample(1:2, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 10)

  expect_error(
    marginal_effects(fit, X),
    "All covariates are factors"
  )
})

# ---- Edge cases --------------------------------------------------------------

test_that("single-level factor works", {
  set.seed(1)
  n <- 50
  X <- data.frame(
    x1 = rnorm(n),
    f = factor(rep("only_level", n))
  )
  Y <- sample(1:2, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 10)
  expect_equal(ncol(fit$predictions), 2)
})

test_that("two-level factor works", {
  set.seed(1)
  n <- 50
  X <- data.frame(x = factor(sample(c("lo", "hi"), n, replace = TRUE)))
  Y <- sample(1:2, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 10)
  expect_true(all(abs(rowSums(fit$predictions) - 1) < 1e-10))
})

test_that("many-level factor (10 levels) works", {
  set.seed(1)
  n <- 200
  X <- data.frame(x = factor(sample(paste0("lv", 1:10), n, replace = TRUE)))
  Y <- sample(1:3, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 10)
  expect_equal(ncol(fit$predictions), 3)
  expect_true(all(abs(rowSums(fit$predictions) - 1) < 1e-10))
})

test_that("character column in validation gives error", {
  X <- data.frame(x = c("a", "b", "c"), stringsAsFactors = FALSE)
  Y <- 1:3
  expect_error(jocf(Y, X), "character.*Convert to factor")
})

test_that("prediction with character input for factor column works", {
  set.seed(1)
  n <- 60
  X <- data.frame(x = factor(sample(c("A", "B"), n, replace = TRUE)))
  Y <- sample(1:2, n, replace = TRUE)
  fit <- jocf(Y, X, num.trees = 10)

  # Character vector (not factor) should also work in prediction
  X_new <- data.frame(x = c("A", "B"))
  preds <- predict(fit, X_new)
  expect_equal(nrow(preds$probabilities), 2)
})
