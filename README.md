# jocf

Joint Ordered Correlation Forest — estimates conditional choice probabilities for ordered discrete outcomes using a single random forest.

## Overview

`jocf` implements the **joint ordered correlation forest** estimator for
conditional choice probabilities of ordered outcomes
*Y* in {1, ..., *M*}. It grows a **single** random forest whose
splitting criterion jointly minimises average estimation error across all *M*
classes simultaneously. This is in contrast to the published
[`ocf`](https://CRAN.R-project.org/package=ocf) package, which grows *M*
separate forests.

The joint splitting rule is algebraically equivalent to average CART Gini
impurity across classes — see `vignette("jocf-theory")` for the derivation.

## Installation

Install the development version from GitHub:

```r
# install.packages("remotes")
remotes::install_github("riccardo-df/jocf")
```

## Quick example

```r
library(jocf)

## Simulate ordered outcome data
set.seed(42)
n <- 500
X <- matrix(rnorm(n * 5), ncol = 5)
Y <- sample(1:4, n, replace = TRUE, prob = c(0.2, 0.3, 0.3, 0.2))

## Fit a joint ordered correlation forest
fit <- jocf(Y, X, num.trees = 500)

## In-sample predicted probabilities (n x M matrix)
head(fit$predictions)

## In-sample classifications
table(fit$classification$prob)   # probability-based
table(fit$classification$vote)   # majority-vote (unique to unified OCF)

## Out-of-sample predictions
X_new <- matrix(rnorm(10 * 5), ncol = 5)
preds <- predict(fit, X_new)
preds$probabilities            # (n_new x M) probability matrix
preds$classification$prob      # probability-based classification
preds$classification$vote      # majority-vote classification

## Average marginal effects
me <- marginal_effects(fit)
me
```

## Key features

- **Joint splitting criterion**: one forest for all *M* classes, equivalent to average Gini impurity.
- **Two classification rules**: probability-based (argmax of averaged probabilities) and majority-vote (per-tree argmax aggregation, unique to the unified OCF).
- **Honest forests with inference**: `honesty = TRUE` enables sample splitting — trees are grown on one subsample, leaves repopulated from a held-out honesty sample. `predict(..., variance = TRUE)` provides weight-based standard errors for each predicted probability; `marginal_effects()` returns SEs automatically for honest models.
- **Weighted splitting**: optional variance-weighted criterion (`splitting.rule = "weighted"`) to equalise contribution of rare classes.
- **Built-in hyperparameter tuning**: GRF-style tuning of `mtry`, `min.node.size`, and `sample.fraction` via debiased OOB error and a Kriging surrogate (`tune.parameters = "all"`).
- **OpenMP parallelism**: tree growing, prediction, and marginal effects are parallelised via OpenMP.
- **Nonparametric marginal effects**: finite-difference estimator with support for continuous and discrete covariates.
