## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)


## ----fit----------------------------------------------------------------------
library(jocf)

## Simulate ordered outcome data
set.seed(123)
n <- 200
X <- matrix(rnorm(n * 4), ncol = 4, dimnames = list(NULL, paste0("x", 1:4)))
latent <- X[, 1] - 0.5 * X[, 2] + rnorm(n)
Y <- as.integer(cut(latent, breaks = c(-Inf, -0.5, 0.5, Inf)))

## Fit a joint ordered correlation forest
fit <- jocf(Y, X, num.trees = 100)


## ----insample-----------------------------------------------------------------
head(fit$predictions)

## Rows sum to 1
summary(rowSums(fit$predictions))


## ----predict------------------------------------------------------------------
X_new <- matrix(rnorm(10 * 4), ncol = 4)
preds <- predict(fit, X_new)

## Predicted probabilities (n_new x M matrix)
preds$probabilities

## Probability-based classification (argmax of averaged probs)
preds$classification$prob

## Majority-vote classification (unique to unified OCF)
preds$classification$vote


## ----me-----------------------------------------------------------------------
## AME averaged over all training observations
me <- marginal_effects(fit, X)
me


## ----me_options---------------------------------------------------------------
## AME evaluated at the covariate means, for covariates 1 and 3 only
me_atmean <- marginal_effects(fit, X, eval = "atmean",
                              target_covariates = c(1, 3))
me_atmean


## ----me_discrete--------------------------------------------------------------
## Treat covariate 4 as discrete
me_disc <- marginal_effects(fit, X, discrete_vars = 4)
me_disc


## ----weighted-----------------------------------------------------------------
fit_w <- jocf(Y, X, num.trees = 100, splitting.rule = "weighted")
head(fit_w$predictions)


## ----threads, eval = FALSE----------------------------------------------------
# ## Use 2 threads
# fit2 <- jocf(Y, X, num.trees = 100, num.threads = 2)
# preds2 <- predict(fit2, X_new, num.threads = 2)
# me2 <- marginal_effects(fit2, X, num.threads = 2)

