# jocf 0.0.0.9000

* **Marginal effects refinements**: renamed `omega` → `bandwidth`, renamed
  `$AME` → `$effects`, added `$ci.lower` / `$ci.upper` (95% CIs) alongside
  `$std.error` for honest models.  Print method adapts header to eval mode.
* **Honesty and inference** (Phase 4): `jocf(..., honesty = TRUE)` grows trees
  on a training subsample and repopulates leaves from a held-out honesty sample.
  - `predict.jocf(..., variance = TRUE)` computes weight-based variance
    estimates and standard errors for each predicted probability.
  - `marginal_effects.jocf()` returns `$std.error`, `$ci.lower`, `$ci.upper`
    for marginal effects when the model is honest.
  - New argument `honesty.fraction` (default 0.5) controls the train/honesty split.
* **Built-in hyperparameter tuning**: `jocf(..., tune.parameters = "all")`
  tunes `mtry`, `min.node.size`, and `sample.fraction` via GRF-style debiased
  OOB error and a Kriging surrogate (`DiceKriging`).  New arguments:
  `tune.parameters`, `tune.num.trees`, `tune.num.reps`, `tune.num.draws`.
  Tuning results stored in `$tuning.output`.
* Initial implementation of the joint ordered correlation forest estimator.
* Core functions: `jocf()`, `predict.jocf()`, `marginal_effects.jocf()`.
* Joint splitting criterion: unweighted (`"simple"`) and variance-weighted
  (`"weighted"`).
* **Classification**: two classification rules derived from the forest
  probability estimates.
    - Probability-based: argmax of forest-averaged probabilities.
    - Majority-vote: each tree votes for its leaf argmax, then aggregate across
      trees.  Unique to the unified OCF (requires a single shared leaf structure
      across all classes).
* `jocf()` returns `$predictions` (probability matrix) and `$classification`
  (list with `$prob` and `$vote` integer vectors).
* `predict.jocf()` returns a named list with `$probabilities` and
  `$classification` (same structure as above).
* C++ backend with OpenMP parallelism for tree growing, prediction, and
  marginal effects.
* Performance optimisations: Armadillo-free hot paths, global pre-sort, BFS
  tree growing, in-place partition, ranked split search.
* **Factor / logical covariate support**: `jocf()` and `predict.jocf()` now
  accept factor (ordered and unordered), logical, and numeric columns.
  Unordered factor levels are sorted by `mean(Y)` and encoded as integer codes;
  ordered factors use their existing level order; logicals map to 1/2.
  `marginal_effects()` auto-excludes non-numeric covariates.
