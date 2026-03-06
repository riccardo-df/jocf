#' @description
#' The \pkg{jocf} package estimates conditional choice probabilities and
#' classifications for ordered discrete outcomes
#' \eqn{Y \in \{1, \ldots, M\}} using a single random forest with a joint
#' splitting criterion.  Unlike the published \pkg{ocf} package, which grows
#' \eqn{M} separate forests, \pkg{jocf} grows one forest whose splitting rule
#' is equivalent to minimising total CART Gini impurity (Proposition 1 in
#' `vignette("jocf-theory")`).  The single-forest structure additionally
#' enables a majority-vote classifier that is not available to multi-forest
#' approaches.
#'
#' @seealso
#' * [jocf()] to fit a forest and obtain in-sample predictions and
#'   classifications
#' * [predict.jocf()] for out-of-sample predictions and classifications
#' * [marginal_effects.jocf()] for nonparametric marginal effects
#'
#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @useDynLib jocf, .registration = TRUE
#' @importFrom Rcpp evalCpp
## usethis namespace: end
NULL
