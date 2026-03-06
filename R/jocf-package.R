#' @description
#' The \pkg{jocf} package estimates conditional choice probabilities for ordered
#' discrete outcomes \eqn{Y \in \{1, \ldots, M\}} using a single random forest
#' with a joint splitting criterion.  Unlike the published \pkg{ocf} package,
#' which grows \eqn{M} separate forests, \pkg{jocf} grows one forest whose
#' splitting rule is equivalent to minimising total CART Gini impurity
#' (Proposition 1 in `vignette("jocf-theory")`).
#'
#' @seealso
#' * [jocf()] to fit a forest
#' * [predict.jocf()] for out-of-sample predictions
#' * [marginal_effects.jocf()] for nonparametric marginal effects
#'
#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @useDynLib jocf, .registration = TRUE
#' @importFrom Rcpp evalCpp
## usethis namespace: end
NULL
