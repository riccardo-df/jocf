# =============================================================================
# Verify jocf splitting + predictions against a pure base-R implementation.
# No C++ knowledge needed — everything here is readable R.
# =============================================================================

devtools::load_all()
set.seed(42)

# --- Data --------------------------------------------------------------------
n <- 200
k <- 3
M <- 3
X <- matrix(rnorm(n * k), ncol = k)
latent <- X[, 1] - 0.5 * X[, 2] + rnorm(n)
Y <- as.integer(cut(latent, breaks = c(-Inf, -0.5, 0.5, Inf)))
min_node_size <- 5
lambda <- rep(1, M)  # unweighted

# =============================================================================
# PART 1: Verify the splitting criterion
# =============================================================================

# --- Base R: exhaustive best-split search ------------------------------------
# Q(C) = (1/M) * sum_m lambda_m * p_m * (1 - p_m)
node_impurity_R <- function(y, M, lambda) {
  n <- length(y)
  if (n == 0) return(0)
  counts <- tabulate(y, nbins = M)
  p <- counts / n
  sum(lambda * p * (1 - p)) / M
}

find_best_split_R <- function(Y, X, M, lambda, min_node_size) {
  n <- length(Y)
  k <- ncol(X)
  best_imp <- Inf
  best_feature <- NA
  best_threshold <- NA

  for (j in seq_len(k)) {
    # Sort observations by feature j
    ord <- order(X[, j])
    x_sorted <- X[ord, j]
    y_sorted <- Y[ord]

    # Sweep through candidate thresholds (midpoints between consecutive
    # distinct values), enforcing min_node_size in each child.
    for (i in seq_len(n - 1)) {
      # Skip ties — no split between identical values
      if (x_sorted[i] == x_sorted[i + 1]) next

      n_left <- i
      n_right <- n - i

      # Enforce min_node_size
      if (n_left < min_node_size || n_right < min_node_size) next

      y_left <- y_sorted[1:i]
      y_right <- y_sorted[(i + 1):n]

      imp <- node_impurity_R(y_left, M, lambda) +
             node_impurity_R(y_right, M, lambda)

      if (imp < best_imp) {
        best_imp <- imp
        best_feature <- j
        best_threshold <- (x_sorted[i] + x_sorted[i + 1]) / 2
      }
    }
  }

  list(feature = best_feature, threshold = best_threshold, impurity = best_imp)
}

# --- Run both ----------------------------------------------------------------
result_R   <- find_best_split_R(Y, X, M, lambda, min_node_size)
result_cpp <- find_best_split_cpp(Y, X, M, lambda, min_node_size)

cat("=== PART 1: Splitting criterion ===\n\n")
cat("Base R result:\n")
cat(sprintf("  Feature:   %d\n", result_R$feature))
cat(sprintf("  Threshold: %.10f\n", result_R$threshold))
cat(sprintf("  Impurity:  %.10f\n\n", result_R$impurity))

cat("C++ result:\n")
cat(sprintf("  Feature:   %d\n", result_cpp$feature))
cat(sprintf("  Threshold: %.10f\n", result_cpp$threshold))
cat(sprintf("  Impurity:  %.10f\n\n", result_cpp$impurity))

cat(sprintf("Feature match:   %s\n", result_R$feature == result_cpp$feature))
cat(sprintf("Threshold match: %s (diff = %.2e)\n",
            abs(result_R$threshold - result_cpp$threshold) < 1e-12,
            abs(result_R$threshold - result_cpp$threshold)))
cat(sprintf("Impurity match:  %s (diff = %.2e)\n\n",
            abs(result_R$impurity - result_cpp$impurity) < 1e-12,
            abs(result_R$impurity - result_cpp$impurity)))


# =============================================================================
# PART 2: Verify predictions from a 1-tree stump
# =============================================================================

# Force a stump: min.node.size large enough that children can't split further.
# With min.node.size >= ceil(n/2), the root can split once but children cannot.
fit <- jocf(Y, X, num.trees = 1, mtry = k,
            min.node.size = ceiling(n / 2))

# Extract the tree structure
tree <- fit$forest[[1]]

cat("=== PART 2: Predictions from a 1-tree stump ===\n\n")
cat("Tree structure:\n")
cat(sprintf("  Number of nodes: %d\n", length(tree$split_feature)))
cat(sprintf("  Split feature:   %d\n", tree$split_feature[1]))
cat(sprintf("  Split threshold: %.10f\n\n", tree$split_threshold[1]))

# --- Predict on new data: C++ vs base R -------------------------------------
X_new <- matrix(rnorm(20 * k), ncol = k)

# C++ predictions
preds_cpp <- predict(fit, X_new)$probabilities

# Base R predictions: manually traverse the stump
split_feat <- tree$split_feature[1]   # 1-based feature index
split_thresh <- tree$split_threshold[1]
leaf_probs <- tree$leaf_probs           # (num_nodes x M) matrix

preds_R <- matrix(NA, nrow = nrow(X_new), ncol = M)
for (i in seq_len(nrow(X_new))) {
  node <- 1  # root (1-based)
  while (tree$split_feature[node] != 0) {  # 0 = leaf
    feat <- tree$split_feature[node]
    if (X_new[i, feat] <= tree$split_threshold[node]) {
      node <- tree$left_child[node] + 1   # stored 0-based, convert to 1-based
    } else {
      node <- tree$right_child[node] + 1
    }
  }
  preds_R[i, ] <- leaf_probs[node, ]
}

cat("C++ predictions (first 5 rows):\n")
print(round(preds_cpp[1:5, ], 6))
cat("\nBase R predictions (first 5 rows):\n")
print(round(preds_R[1:5, ], 6))

max_diff <- max(abs(preds_cpp - preds_R))
cat(sprintf("\nMax absolute difference: %.2e\n", max_diff))
cat(sprintf("All predictions match:  %s\n\n", max_diff < 1e-12))

# --- Verify leaf proportions are actual class frequencies --------------------
# We can't access the bootstrap sample, but we CAN verify the leaf probs
# are valid probability vectors.
cat("Leaf probability checks:\n")
for (node in seq_len(nrow(leaf_probs))) {
  if (tree$split_feature[node] == 0) {  # leaf
    cat(sprintf("  Node %d (leaf): probs = [%s], sum = %.10f, all >= 0: %s\n",
                node,
                paste(round(leaf_probs[node, ], 4), collapse = ", "),
                sum(leaf_probs[node, ]),
                all(leaf_probs[node, ] >= 0)))
  }
}

cat("\n=== DONE ===\n")
