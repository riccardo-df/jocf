#' @importFrom stats predict sd median plogis rnorm rbinom rlogis quantile setNames
NULL

#' Resolve num.threads to an integer for C++ (0 = all available cores)
#' @keywords internal
resolve_num_threads <- function(num.threads) {
  if (is.null(num.threads)) return(0L)
  nt <- as.integer(num.threads)
  if (length(nt) != 1L || is.na(nt) || nt < 1L)
    stop("`num.threads` must be a positive integer or NULL.", call. = FALSE)
  nt
}

#' Validate inputs to jocf()
#'
#' @param Y Integer vector of outcome labels.
#' @param X Matrix or data.frame of covariates.
#' @param num.trees Positive integer number of trees.
#' @param min.node.size Positive integer minimum node size.
#' @param max.depth Positive integer or NULL (unlimited).
#' @param sample.fraction Numeric in (0, 1]; subsample fraction.
#' @param mtry Positive integer; feature subsample size.
#' @param splitting.rule Character, one of `"simple"` or `"weighted"`.
#' @param honesty Logical; `TRUE` enables honest splitting.
#' @param honesty.fraction Numeric in (0, 1); fraction allocated to honesty
#'   sample.  Only validated when `honesty = TRUE`.
#'
#' @return Invisibly returns `NULL`. Stops with an informative message on
#'   invalid input.
#' @keywords internal
validate_jocf_inputs <- function(Y, X, num.trees, min.node.size, max.depth,
                                  sample.fraction, mtry, splitting.rule,
                                  honesty, honesty.fraction = 0.5) {
  if (!is.integer(Y) && !all(Y == floor(Y)))
    stop("`Y` must be an integer vector.", call. = FALSE)
  Y <- as.integer(Y)
  M <- max(Y)
  if (min(Y) < 1L)
    stop("`Y` values must be >= 1.", call. = FALSE)
  if (!all(Y %in% seq_len(M)))
    stop("`Y` must contain consecutive integer values starting at 1.",
         call. = FALSE)

  if (!is.matrix(X) && !is.data.frame(X))
    stop("`X` must be a matrix or data.frame.", call. = FALSE)
  if (is.matrix(X) && !is.numeric(X))
    stop("`X` must be numeric (or a data.frame with factor/logical columns).",
         call. = FALSE)
  if (is.data.frame(X)) {
    col_classes <- vapply(X, function(col) {
      if (is.numeric(col)) return("numeric")
      if (is.logical(col)) return("logical")
      if (is.factor(col))  return("factor")
      if (is.character(col)) return("character")
      class(col)[1]
    }, character(1))
    bad <- col_classes == "character"
    if (any(bad))
      stop(sprintf("Column(s) %s are character. Convert to factor first.",
                   paste(which(bad), collapse = ", ")), call. = FALSE)
    unsupported <- !col_classes %in% c("numeric", "logical", "factor")
    if (any(unsupported))
      stop(sprintf("Column(s) %s have unsupported type(s).",
                   paste(which(unsupported), collapse = ", ")), call. = FALSE)
  }
  if (nrow(X) != length(Y))
    stop("`X` and `Y` must have the same number of rows.", call. = FALSE)
  if (anyNA(Y))
    stop("Missing values are not allowed in `Y`.", call. = FALSE)
  if (anyNA(if (is.data.frame(X)) do.call(cbind, lapply(X, as.numeric)) else X))
    stop("Missing values are not allowed in `X`.", call. = FALSE)

  if (!is.numeric(num.trees) || length(num.trees) != 1L || num.trees < 1L)
    stop("`num.trees` must be a positive integer.", call. = FALSE)
  if (!is.numeric(min.node.size) || length(min.node.size) != 1L ||
      min.node.size < 1L)
    stop("`min.node.size` must be a positive integer.", call. = FALSE)
  if (!is.null(max.depth)) {
    if (!is.numeric(max.depth) || length(max.depth) != 1L ||
        max.depth < 1L || max.depth != floor(max.depth))
      stop("`max.depth` must be a positive integer or NULL.", call. = FALSE)
  }
  if (!is.numeric(sample.fraction) || length(sample.fraction) != 1L ||
      sample.fraction <= 0 || sample.fraction > 1)
    stop("`sample.fraction` must be a number in (0, 1].", call. = FALSE)
  if (!is.null(mtry)) {
    if (!is.numeric(mtry) || length(mtry) != 1L || mtry < 1L ||
        mtry > ncol(X))
      stop("`mtry` must be a positive integer <= ncol(X).", call. = FALSE)
  }
  if (!splitting.rule %in% c("simple", "weighted"))
    stop('`splitting.rule` must be "simple" or "weighted".', call. = FALSE)
  if (!is.logical(honesty) || length(honesty) != 1L)
    stop("`honesty` must be TRUE or FALSE.", call. = FALSE)
  if (isTRUE(honesty)) {
    if (!is.numeric(honesty.fraction) || length(honesty.fraction) != 1L ||
        honesty.fraction <= 0 || honesty.fraction >= 1)
      stop("`honesty.fraction` must be a number in (0, 1).", call. = FALSE)
  }

  invisible(NULL)
}


#' Compute variance-weighting lambdas from global class proportions
#'
#' @param Y Integer vector of class labels (1..M).
#' @param M Number of classes.
#' @return Numeric vector of length M with lambda_m = 1 / (p_m * (1 - p_m)).
#' @keywords internal
compute_lambda <- function(Y, M) {
  n <- length(Y)
  p_hat <- tabulate(Y, nbins = M) / n
  # Guard against extreme proportions to avoid division by zero
  p_hat <- pmax(p_hat, 1e-10)
  p_hat <- pmin(p_hat, 1 - 1e-10)
  1.0 / (p_hat * (1.0 - p_hat))
}
