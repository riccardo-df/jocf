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


## ----classification-----------------------------------------------------------
## In-sample classifications
table(fit$classification$prob)
table(fit$classification$vote)


## ----predict------------------------------------------------------------------
X_new <- matrix(rnorm(10 * 4), ncol = 4)
preds <- predict(fit, X_new)

## Predicted probabilities (n_new x M matrix)
head(preds$probabilities)

## Probability-based classification
preds$classification$prob

## Majority-vote classification
preds$classification$vote


## ----me-----------------------------------------------------------------------
## Marginal effects averaged over all training observations
me <- marginal_effects(fit)
me


## ----me_options---------------------------------------------------------------
## Marginal effects evaluated at the covariate means, for x1 and x3 only
me_atmean <- marginal_effects(fit, eval = "atmean",
                              target_covariates = c(x1 = "continuous",
                                                    x3 = "continuous"))
me_atmean


## ----me_discrete--------------------------------------------------------------
## Treat covariate x4 as discrete
me_disc <- marginal_effects(fit,
                            target_covariates = c(x4 = "discrete"))
me_disc


## ----weighted-----------------------------------------------------------------
fit_w <- jocf(Y, X, num.trees = 100, splitting.rule = "weighted")
head(fit_w$predictions)


## ----threads, eval = FALSE----------------------------------------------------
# ## Use 2 threads
# fit2 <- jocf(Y, X, num.trees = 100, num.threads = 2)
# preds2 <- predict(fit2, X_new, num.threads = 2)
# me2 <- marginal_effects(fit2, num.threads = 2)

