#' Encode factor, ordered factor, and logical columns to numeric codes
#'
#' In **training mode** (`Y` supplied, `factor_info = NULL`), detects non-numeric
#' columns in `X`, converts them to integer codes, and returns the encoding
#' metadata so it can be reapplied at prediction time.
#'
#' In **prediction mode** (`factor_info` supplied), re-encodes `X` using the
#' stored level orderings from training.
#'
#' @param X A data.frame or matrix of covariates.
#' @param Y Integer vector of outcomes (training mode only). Used to order
#'   unordered factor levels by `mean(Y)`.
#' @param factor_info List returned by a previous `encode_factors()` call
#'   (prediction mode). `NULL` in training mode.
#'
#' @return A list with components:
#' \describe{
#'   \item{`X_encoded`}{Numeric matrix with the same dimensions as `X`.}
#'   \item{`factor_info`}{Named list (one entry per column, `NULL` for numeric
#'     columns) storing the encoding metadata. `NULL` when `X` had no
#'     non-numeric columns.}
#' }
#' @keywords internal
encode_factors <- function(X, Y = NULL, factor_info = NULL) {

  # --- Training mode ----------------------------------------------------------
  if (is.null(factor_info)) {
    # Fast path: already a numeric matrix → nothing to do
    if (is.matrix(X) && is.numeric(X))
      return(list(X_encoded = X, factor_info = NULL))

    X <- as.data.frame(X)
    k <- ncol(X)
    fi <- vector("list", k)
    names(fi) <- colnames(X)
    any_encoded <- FALSE

    X_out <- matrix(NA_real_, nrow = nrow(X), ncol = k)
    colnames(X_out) <- colnames(X)

    for (j in seq_len(k)) {
      col <- X[[j]]

      if (is.numeric(col)) {
        X_out[, j] <- col
        fi[j] <- list(NULL)

      } else if (is.logical(col)) {
        X_out[, j] <- ifelse(col, 2L, 1L)
        fi[[j]] <- list(type = "logical", levels = c("FALSE", "TRUE"),
                        codes = 1:2)
        any_encoded <- TRUE

      } else if (is.ordered(col)) {
        lvls <- levels(col)
        X_out[, j] <- as.integer(col)
        fi[[j]] <- list(type = "ordered", levels = lvls,
                        codes = seq_along(lvls))
        any_encoded <- TRUE

      } else if (is.factor(col)) {
        lvls <- levels(col)
        # Order levels by mean(Y) — natural for ordered outcomes
        means <- vapply(lvls, function(lv) mean(Y[col == lv]), numeric(1))
        ord   <- order(means)
        lvls_sorted <- lvls[ord]
        # Map: original level → code
        code_map <- integer(length(lvls))
        code_map[ord] <- seq_along(lvls)
        X_out[, j] <- code_map[as.integer(col)]
        fi[[j]] <- list(type = "unordered", levels = lvls_sorted,
                        codes = seq_along(lvls_sorted))
        any_encoded <- TRUE

      } else if (is.character(col)) {
        stop(sprintf(
          "Column %d (\"%s\") is character. Convert to factor first.",
          j, colnames(X)[j]), call. = FALSE)

      } else {
        stop(sprintf(
          "Column %d (\"%s\") has unsupported type \"%s\".",
          j, colnames(X)[j], class(col)[1]), call. = FALSE)
      }
    }

    if (!any_encoded) fi <- NULL
    return(list(X_encoded = X_out, factor_info = fi))
  }

  # --- Prediction mode --------------------------------------------------------
  # factor_info is NULL → pure numeric training data, just coerce

  if (is.null(factor_info)) {
    X <- as.matrix(X)
    if (!is.numeric(X))
      stop("`newdata` must be numeric (training data had no factor columns).",
           call. = FALSE)
    return(list(X_encoded = X, factor_info = NULL))
  }

  X <- as.data.frame(X)
  k <- ncol(X)

  if (length(factor_info) != k)
    stop(sprintf("`newdata` must have %d column(s) (same as training X).",
                 length(factor_info)), call. = FALSE)

  X_out <- matrix(NA_real_, nrow = nrow(X), ncol = k)
  colnames(X_out) <- colnames(X)

  for (j in seq_len(k)) {
    col <- X[[j]]
    info <- factor_info[[j]]

    if (is.null(info)) {
      # Numeric column at training time
      if (!is.numeric(col))
        stop(sprintf(
          "Column %d (\"%s\"): training had numeric, got %s.",
          j, colnames(X)[j], class(col)[1]), call. = FALSE)
      X_out[, j] <- col

    } else if (info$type == "logical") {
      if (!is.logical(col))
        stop(sprintf(
          "Column %d (\"%s\"): training had logical, got %s.",
          j, colnames(X)[j], class(col)[1]), call. = FALSE)
      X_out[, j] <- ifelse(col, 2L, 1L)

    } else if (info$type %in% c("ordered", "unordered")) {
      if (!is.factor(col) && !is.character(col))
        stop(sprintf(
          "Column %d (\"%s\"): training had factor, got %s.",
          j, colnames(X)[j], class(col)[1]), call. = FALSE)
      col_char <- as.character(col)
      stored_levels <- info$levels
      codes <- match(col_char, stored_levels)
      # Unseen levels → warning + median code
      unseen <- is.na(codes) & !is.na(col_char)
      if (any(unseen)) {
        unseen_lvls <- unique(col_char[unseen])
        warning(sprintf(
          "Column %d (\"%s\"): unseen level(s) %s mapped to median code.",
          j, colnames(X)[j],
          paste0("\"", unseen_lvls, "\"", collapse = ", ")),
          call. = FALSE)
        codes[unseen] <- ceiling(length(stored_levels) / 2)
      }
      X_out[, j] <- codes
    }
  }

  list(X_encoded = X_out, factor_info = factor_info)
}
