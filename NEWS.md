# jocf 0.0.0.9000

* Initial implementation of the joint ordered correlation forest estimator.
* Core functions: `jocf()`, `predict.jocf()`, `marginal_effects.jocf()`.
* Joint splitting criterion: unweighted (`"simple"`) and variance-weighted (`"weighted"`).
* C++ backend with OpenMP parallelism for tree growing, prediction, and marginal effects.
* Performance optimisations: Armadillo-free hot paths, global pre-sort, BFS tree growing, in-place partition, ranked split search.
