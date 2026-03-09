## Tests for generate_ordered_data()

test_that("output structure is correct (default M=3)", {
  set.seed(1)
  d <- generate_ordered_data(200)

  expect_type(d, "list")
  expect_named(d, c("sample", "true_probs", "marginal_probs",
                     "true_me_atmean", "true_me_atmedian"))
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


## --- True marginal effects tests -------------------------------------------

test_that("true_me fields are present in output", {
  set.seed(30)
  d <- generate_ordered_data(200)
  expect_true("true_me_atmean" %in% names(d))
  expect_true("true_me_atmedian" %in% names(d))
})

test_that("true_me dimensions are (6 x M) for default M=3", {
  set.seed(31)
  d <- generate_ordered_data(200)
  expect_equal(dim(d$true_me_atmean), c(6, 3))
  expect_equal(dim(d$true_me_atmedian), c(6, 3))
})

test_that("true_me dimensions are (6 x M) for M=2", {
  set.seed(32)
  d <- generate_ordered_data(200, n_categories = 2)
  expect_equal(dim(d$true_me_atmean), c(6, 2))
  expect_equal(dim(d$true_me_atmedian), c(6, 2))
})

test_that("true_me dimensions are (6 x M) for M=4", {
  set.seed(33)
  d <- generate_ordered_data(200, n_categories = 4)
  expect_equal(dim(d$true_me_atmean), c(6, 4))
  expect_equal(dim(d$true_me_atmedian), c(6, 4))
})

test_that("true_me dimensions are (6 x M) for M=5", {
  set.seed(34)
  d <- generate_ordered_data(200, n_categories = 5)
  expect_equal(dim(d$true_me_atmean), c(6, 5))
  expect_equal(dim(d$true_me_atmedian), c(6, 5))
})

test_that("true_me row sums are zero (probabilities sum to 1)", {
  set.seed(35)
  d <- generate_ordered_data(500)
  expect_equal(unname(rowSums(d$true_me_atmean)), rep(0, 6), tolerance = 1e-10)
  expect_equal(unname(rowSums(d$true_me_atmedian)), rep(0, 6), tolerance = 1e-10)
})

test_that("true_me row sums are zero for M=5", {
  set.seed(36)
  d <- generate_ordered_data(500, n_categories = 5)
  expect_equal(unname(rowSums(d$true_me_atmean)), rep(0, 6), tolerance = 1e-10)
  expect_equal(unname(rowSums(d$true_me_atmedian)), rep(0, 6), tolerance = 1e-10)
})

test_that("noise covariates (x5, x6) have exactly zero MEs", {
  set.seed(37)
  d <- generate_ordered_data(300)
  expect_equal(as.numeric(d$true_me_atmean[5, ]), rep(0, 3))
  expect_equal(as.numeric(d$true_me_atmean[6, ]), rep(0, 3))
  expect_equal(as.numeric(d$true_me_atmedian[5, ]), rep(0, 3))
  expect_equal(as.numeric(d$true_me_atmedian[6, ]), rep(0, 3))
})

test_that("noise covariates zero for M=5", {
  set.seed(38)
  d <- generate_ordered_data(300, n_categories = 5)
  expect_equal(as.numeric(d$true_me_atmean[5, ]), rep(0, 5))
  expect_equal(as.numeric(d$true_me_atmean[6, ]), rep(0, 5))
})

test_that("true_me has correct row and column names", {
  set.seed(39)
  d <- generate_ordered_data(200)
  expect_equal(rownames(d$true_me_atmean), paste0("x", 1:6))
  expect_equal(colnames(d$true_me_atmean), paste0("Y", 1:3))
  expect_equal(rownames(d$true_me_atmedian), paste0("x", 1:6))
  expect_equal(colnames(d$true_me_atmedian), paste0("Y", 1:3))
})

test_that("true_me names correct for M=5", {
  set.seed(40)
  d <- generate_ordered_data(200, n_categories = 5)
  expect_equal(colnames(d$true_me_atmean), paste0("Y", 1:5))
  expect_equal(colnames(d$true_me_atmedian), paste0("Y", 1:5))
})

test_that("active covariates have nonzero MEs", {
  set.seed(41)
  d <- generate_ordered_data(500)
  # x1 (continuous, beta=1) and x3 (continuous, beta=0.5) should be nonzero
  expect_true(any(d$true_me_atmean[1, ] != 0))
  expect_true(any(d$true_me_atmean[3, ] != 0))
  # x2 (discrete, beta=1) and x4 (discrete, beta=0.5) should be nonzero
  expect_true(any(d$true_me_atmean[2, ] != 0))
  expect_true(any(d$true_me_atmean[4, ] != 0))
})

test_that("continuous ME magnitudes scale with beta", {
  set.seed(42)
  d <- generate_ordered_data(500)
  # |ME(x1)| / |ME(x3)| should be close to beta1/beta3 = 1/0.5 = 2
  # for continuous covariates the ratio is exact
  me1 <- d$true_me_atmean[1, ]
  me3 <- d$true_me_atmean[3, ]
  expect_equal(unname(me1 / me3), rep(2, 3), tolerance = 1e-10)
})

test_that("sign pattern: highest class ME positive for positive beta continuous covariate", {
  set.seed(43)
  d <- generate_ordered_data(500)
  M <- 3
  # For positive beta, increasing x pushes Y up, so ME for highest class > 0
  expect_true(d$true_me_atmean[1, M] > 0)
  expect_true(d$true_me_atmean[3, M] > 0)
})

test_that("true_me values are finite", {
  set.seed(44)
  d <- generate_ordered_data(200)
  expect_true(all(is.finite(d$true_me_atmean)))
  expect_true(all(is.finite(d$true_me_atmedian)))
})
