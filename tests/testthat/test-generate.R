## Tests for generate_ordered_data()

test_that("output structure is correct (default M=3)", {
  set.seed(1)
  d <- generate_ordered_data(200)

  expect_type(d, "list")
  expect_named(d, c("sample", "true_probs", "marginal_probs"))
  expect_s3_class(d$sample, "data.frame")
  expect_true(is.matrix(d$true_probs))
  expect_true(is.numeric(d$marginal_probs))
})

test_that("dimensions are correct", {
  set.seed(2)
  n <- 150
  d <- generate_ordered_data(n)
  M <- 3

  expect_equal(nrow(d$sample), n)
  expect_equal(ncol(d$sample), 7)
  expect_equal(colnames(d$sample), c("Y", paste0("x", 1:6)))
  expect_equal(nrow(d$true_probs), n)
  expect_equal(ncol(d$true_probs), M)
  expect_equal(length(d$marginal_probs), M)
})

test_that("Y values are in 1..M and all classes present for large n", {
  set.seed(3)
  d <- generate_ordered_data(5000)
  Y <- d$sample$Y

  expect_true(all(Y %in% 1:3))
  expect_length(unique(Y), 3)
})

test_that("true_probs rows sum to 1 and are in [0,1]", {
  set.seed(4)
  d <- generate_ordered_data(300)

  expect_equal(rowSums(d$true_probs), rep(1, 300), tolerance = 1e-10)
  expect_true(all(d$true_probs >= 0))
  expect_true(all(d$true_probs <= 1))
})

test_that("class proportions roughly match marginal_probs for large n", {
  set.seed(5)
  d <- generate_ordered_data(20000)
  Y <- d$sample$Y
  obs_props <- tabulate(Y, nbins = 3) / length(Y)

  # Default is c(1/3, 1/3, 1/3)
  expect_equal(obs_props, rep(1/3, 3), tolerance = 0.05)
})

test_that("imbalanced marginal_probs produces skewed distribution", {
  set.seed(6)
  probs <- c(0.8, 0.1, 0.1)
  d <- generate_ordered_data(20000, marginal_probs = probs)
  Y <- d$sample$Y
  obs_props <- tabulate(Y, nbins = 3) / length(Y)

  expect_equal(obs_props, probs, tolerance = 0.05)
})

test_that("n_categories = 4 with default probs works", {
  set.seed(7)
  d <- generate_ordered_data(5000, n_categories = 4)

  expect_equal(ncol(d$true_probs), 4)
  expect_equal(length(d$marginal_probs), 4)
  expect_true(all(d$sample$Y %in% 1:4))
  expect_length(unique(d$sample$Y), 4)
  expect_equal(rowSums(d$true_probs), rep(1, 5000), tolerance = 1e-10)
})

test_that("n_categories and marginal_probs length must be consistent", {
  expect_error(
    generate_ordered_data(100, n_categories = 3, marginal_probs = c(0.5, 0.5)),
    "length equal to `n_categories`"
  )
})

test_that("rejects invalid n", {
  expect_error(generate_ordered_data(0), "positive integer")
  expect_error(generate_ordered_data(-5), "positive integer")
  expect_error(generate_ordered_data(2.5), "positive integer")
  expect_error(generate_ordered_data("a"), "positive integer")
})

test_that("rejects invalid n_categories", {
  expect_error(generate_ordered_data(100, n_categories = 1), ">= 2")
  expect_error(generate_ordered_data(100, n_categories = 2.5), ">= 2")
})

test_that("rejects invalid marginal_probs", {
  expect_error(
    generate_ordered_data(100, marginal_probs = c(0.5, 0.3, 0.1)),
    "sum to 1"
  )
  expect_error(
    generate_ordered_data(100, marginal_probs = c(-0.1, 0.6, 0.5)),
    "positive numeric"
  )
  expect_error(
    generate_ordered_data(100, marginal_probs = c(0, 0.5, 0.5)),
    "positive numeric"
  )
})

test_that("marginal_probs default equals rep(1/M, M)", {
  set.seed(8)
  d <- generate_ordered_data(50)
  expect_equal(d$marginal_probs, rep(1/3, 3))
})

test_that("custom M=5 with explicit probs works", {
  set.seed(9)
  probs <- c(0.1, 0.2, 0.3, 0.25, 0.15)
  d <- generate_ordered_data(10000, n_categories = 5, marginal_probs = probs)

  expect_equal(ncol(d$true_probs), 5)
  expect_true(all(d$sample$Y %in% 1:5))
  expect_equal(rowSums(d$true_probs), rep(1, 10000), tolerance = 1e-10)

  obs_props <- tabulate(d$sample$Y, nbins = 5) / 10000
  expect_equal(obs_props, probs, tolerance = 0.05)
})
