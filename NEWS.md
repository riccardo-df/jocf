# jocf 0.0.0.9000

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
