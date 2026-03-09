// forest.cpp
// Phase 2/3: grow_forest_cpp, predict_forest_cpp, marginal_effects_cpp.
//
// Design:
//   grow_forest_cpp    — 3-phase: (1) sequential R-RNG seeding, (2) parallel
//                        tree growing + prediction accumulation, (3) sequential
//                        serialisation to R lists.
//   predict_forest_cpp — deserialises forest, parallel traverse + accumulate.
//   marginal_effects_cpp — deserialises forest, parallel finite-difference ME.
//
// OpenMP is used when available (_OPENMP defined).  #ifdef guards ensure
// graceful fallback to single-threaded execution otherwise.
//
// Tier 1 optimisation:
//   - X transposed to row-major layout once per function call; prediction
//     loops use const double* xi = x_rowmaj + i*k instead of arma::rowvec.
//   - Per-thread prediction accumulators are flat std::vector<double> (no
//     Armadillo in the inner loop).
//   - marginal_effects_cpp copies rows from row-major layout instead of
//     Armadillo element access.

#include "jocf_internal.h"
#include <cmath>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
// [[Rcpp::depends(RcppArmadillo)]]

// ===========================================================================
// Static helpers (internal linkage)
// ===========================================================================

// ---------------------------------------------------------------------------
// list_to_tree: deserialise a stored R list back to native TreeData.
// Called outside any parallel region (reading R API objects is not thread-safe).
// ---------------------------------------------------------------------------
static TreeData list_to_tree(const Rcpp::List& tree, int M) {
  Rcpp::IntegerVector sf = tree["split_feature"];
  Rcpp::NumericVector st = tree["split_threshold"];
  Rcpp::IntegerVector lc = tree["left_child"];
  Rcpp::IntegerVector rc = tree["right_child"];
  Rcpp::NumericMatrix lp = tree["leaf_probs"];

  const int n_nodes = sf.size();
  TreeData td;
  td.n_nodes = n_nodes;
  td.split_feature.resize(n_nodes);
  td.split_threshold.resize(n_nodes);
  td.left_child.resize(n_nodes);
  td.right_child.resize(n_nodes);
  td.leaf_probs.resize(static_cast<std::size_t>(n_nodes) * M);

  for (int i = 0; i < n_nodes; ++i) {
    td.split_feature[i]   = sf[i];
    td.split_threshold[i] = st[i];
    td.left_child[i]      = lc[i];
    td.right_child[i]     = rc[i];
    for (int m = 0; m < M; ++m)
      td.leaf_probs[i * M + m] = lp(i, m);
  }
  return td;
}

// ---------------------------------------------------------------------------
// tree_to_list: serialise native TreeData to an R list for storage in jocf object.
// ---------------------------------------------------------------------------
static Rcpp::List tree_to_list(const TreeData& td, int M) {
  const int n_nodes = td.n_nodes;
  Rcpp::IntegerVector SF(n_nodes), LC(n_nodes), RC(n_nodes);
  Rcpp::NumericVector ST(n_nodes);
  Rcpp::NumericMatrix LP(n_nodes, M);

  for (int i = 0; i < n_nodes; ++i) {
    SF[i] = td.split_feature[i];
    ST[i] = td.split_threshold[i];
    LC[i] = td.left_child[i];
    RC[i] = td.right_child[i];
    for (int m = 0; m < M; ++m) LP(i, m) = td.leaf_probs[i * M + m];
  }
  return Rcpp::List::create(
    Rcpp::Named("split_feature")   = SF,
    Rcpp::Named("split_threshold") = ST,
    Rcpp::Named("left_child")      = LC,
    Rcpp::Named("right_child")     = RC,
    Rcpp::Named("leaf_probs")      = LP
  );
}

// ---------------------------------------------------------------------------
// traverse_native: walk a native tree to a leaf for observation xi.
// xi must support operator[](int) returning a double.
// Read-only; thread-safe.
// ---------------------------------------------------------------------------
template <typename T>
static int traverse_native(const TreeData& td, const T& xi) {
  int node = 0;
  while (td.split_feature[node] != 0) {
    const int feat = td.split_feature[node] - 1;  // 0-based column
    node = (xi[feat] <= td.split_threshold[node])
               ? td.left_child[node]
               : td.right_child[node];
  }
  return node;
}

// ---------------------------------------------------------------------------
// to_rowmaj: transpose a column-major arma matrix to a flat row-major buffer.
// Returns a std::vector<double> of length n_rows * n_cols where
//   result[i * n_cols + j] = x(i, j).
// Done once per function call; eliminates per-observation arma::rowvec allocation.
// ---------------------------------------------------------------------------
static std::vector<double> to_rowmaj(const arma::mat& x) {
  const int n    = static_cast<int>(x.n_rows);
  const int k    = static_cast<int>(x.n_cols);
  const double* xp = x.memptr();   // column-major: xp[j * n + i] = x(i, j)
  std::vector<double> rm(static_cast<std::size_t>(n) * k);
  for (int j = 0; j < k; ++j) {
    const double* col = xp + static_cast<std::ptrdiff_t>(j) * n;
    for (int i = 0; i < n; ++i)
      rm[static_cast<std::size_t>(i) * k + j] = col[i];
  }
  return rm;
}

// ---------------------------------------------------------------------------
// build_sort_data: construct global pre-sort structure from col-major X.
// Cost: O(k * n * log n). Called once per forest.
// ---------------------------------------------------------------------------
static SortData build_sort_data(const arma::mat& x) {
  const int n = static_cast<int>(x.n_rows);
  const int k = static_cast<int>(x.n_cols);

  SortData sd;
  sd.n    = n;
  sd.k    = k;
  sd.x_cm = x.memptr();
  sd.index_data.resize(static_cast<std::size_t>(k) * n);
  sd.unique_values.resize(k);
  sd.n_unique.resize(k);

  std::vector<std::pair<double, int>> val_idx(n);
  for (int j = 0; j < k; ++j) {
    const double* col = sd.x_cm + static_cast<std::ptrdiff_t>(j) * n;
    for (int i = 0; i < n; ++i) val_idx[i] = {col[i], i};
    std::sort(val_idx.begin(), val_idx.end());

    sd.unique_values[j].clear();
    sd.unique_values[j].push_back(val_idx[0].first);
    int rank = 0;
    sd.index_data[static_cast<std::ptrdiff_t>(j) * n
                   + val_idx[0].second] = 0;
    for (int i = 1; i < n; ++i) {
      if (val_idx[i].first > val_idx[i - 1].first) {
        sd.unique_values[j].push_back(val_idx[i].first);
        ++rank;
      }
      sd.index_data[static_cast<std::ptrdiff_t>(j) * n
                     + val_idx[i].second] = rank;
    }
    sd.n_unique[j] = rank + 1;
  }
  return sd;
}

// ===========================================================================
// grow_forest_cpp (exported)
// ===========================================================================

//' Grow a joint OCF random forest
//'
//' Internal C++ engine called by [jocf()].
//'
//' @param Y Integer vector of class labels (1..M), length n.
//' @param X Numeric matrix of covariates (n x k).
//' @param num_trees Number of trees.
//' @param min_node_size Minimum observations per terminal node.
//' @param max_depth Maximum tree depth (-1 = unlimited).
//' @param n_sub Subsample size (drawn without replacement).
//' @param mtry Number of candidate features at each split.
//' @param M Number of outcome classes.
//' @param lambda Numeric weight vector of length M.
//' @param num_threads Number of OpenMP threads (0 = all available).
//'
//' @return Named list: `predictions` (n x M), `forest` (list of B trees),
//'   `votes` (n x M integer matrix of per-tree majority votes).
//' @keywords internal
//' @export
// [[Rcpp::export]]
Rcpp::List grow_forest_cpp(
  Rcpp::IntegerVector Y,
  Rcpp::NumericMatrix X,
  int                 num_trees,
  int                 min_node_size,
  int                 max_depth,
  int                 n_sub,
  int                 mtry,
  int                 M,
  Rcpp::NumericVector lambda,
  int                 num_threads = 0
) {
  const int n = Y.size();

  // Convert Y to 0-indexed plain int vector
  std::vector<int> y_vec(n);
  for (int i = 0; i < n; ++i) y_vec[i] = Y[i] - 1;

  const arma::mat x = Rcpp::as<arma::mat>(X);
  const int k       = static_cast<int>(x.n_cols);

  // Raw pointer to lambda weights
  const double* lam_raw = lambda.begin();

  // Pre-transpose X to row-major once for in-sample prediction loop.
  // x_rowmaj[i * k + j] = x(i, j).
  const std::vector<double> x_rowmaj = to_rowmaj(x);

  // Phase 0 — Global pre-sort
  SortData sort_data = build_sort_data(x);

  // -------------------------------------------------------------------------
  // Phase 1 — sequential, R's RNG
  // Generate per-tree std::mt19937 seeds and bootstrap index arrays.
  // All R API calls are confined to this phase so the parallel phase is safe.
  // -------------------------------------------------------------------------
  Rcpp::NumericVector seed_doubles = Rcpp::runif(num_trees, 0.0, 4294967295.0);
  std::vector<uint32_t> tree_seeds(num_trees);
  for (int b = 0; b < num_trees; ++b)
    tree_seeds[b] = static_cast<uint32_t>(seed_doubles[b]);

  std::vector<std::vector<int>> boot_indices(num_trees);
  for (int b = 0; b < num_trees; ++b) {
    Rcpp::IntegerVector boot_r = Rcpp::sample(n, n_sub, /*replace=*/false) - 1;
    boot_indices[b].assign(boot_r.begin(), boot_r.end());
  }

  // -------------------------------------------------------------------------
  // Phase 2 — parallel over trees
  // -------------------------------------------------------------------------
  std::vector<TreeData> native_forest(num_trees);
  // Flat row-major prediction accumulator: pred_flat[i * M + m]
  std::vector<double> pred_flat(static_cast<std::size_t>(n) * M, 0.0);
  // Flat row-major vote accumulator: vote_flat[i * M + m] = # trees voting class m for obs i
  std::vector<int> vote_flat(static_cast<std::size_t>(n) * M, 0);

#ifdef _OPENMP
  const int nt_grow = (num_threads > 0) ? num_threads : omp_get_max_threads();
#pragma omp parallel num_threads(nt_grow)
  {
    // Per-thread accumulators
    std::vector<double> local_pred(static_cast<std::size_t>(n) * M, 0.0);
    std::vector<int>    local_vote(static_cast<std::size_t>(n) * M, 0);

#pragma omp for schedule(dynamic, 4)
    for (int b = 0; b < num_trees; ++b) {
      std::mt19937 rng(tree_seeds[b]);
      native_forest[b] = grow_single_tree(
        y_vec.data(), sort_data,
        boot_indices[b].data(), n_sub,
        M, lam_raw, min_node_size, mtry, rng, max_depth
      );
      const TreeData& td   = native_forest[b];
      const double*   xr   = x_rowmaj.data();

      for (int i = 0; i < n; ++i) {
        const double* xi   = xr + static_cast<std::ptrdiff_t>(i) * k;
        const int     leaf = traverse_native(td, xi);
        const double* lp   = td.leaf_probs.data() + leaf * M;
        double*       acc  = local_pred.data() + static_cast<std::ptrdiff_t>(i) * M;
        for (int m = 0; m < M; ++m) acc[m] += lp[m];
        // Majority vote: find argmax of leaf probs for this tree
        int best_m = 0;
        double best_p = lp[0];
        for (int m = 1; m < M; ++m) {
          if (lp[m] > best_p) { best_p = lp[m]; best_m = m; }
        }
        local_vote[static_cast<std::size_t>(i) * M + best_m] += 1;
      }
    }

#pragma omp critical
    {
      for (std::size_t idx = 0; idx < local_pred.size(); ++idx)
        pred_flat[idx] += local_pred[idx];
      for (std::size_t idx = 0; idx < local_vote.size(); ++idx)
        vote_flat[idx] += local_vote[idx];
    }
  }
#else
  for (int b = 0; b < num_trees; ++b) {
    std::mt19937 rng(tree_seeds[b]);
    native_forest[b] = grow_single_tree(
      y_vec.data(), sort_data,
      boot_indices[b].data(), n_sub,
      M, lam_raw, min_node_size, mtry, rng, max_depth
    );
    const TreeData& td   = native_forest[b];
    const double*   xr   = x_rowmaj.data();

    for (int i = 0; i < n; ++i) {
      const double* xi   = xr + static_cast<std::ptrdiff_t>(i) * k;
      const int     leaf = traverse_native(td, xi);
      const double* lp   = td.leaf_probs.data() + leaf * M;
      double*       acc  = pred_flat.data() + static_cast<std::ptrdiff_t>(i) * M;
      for (int m = 0; m < M; ++m) acc[m] += lp[m];
      // Majority vote: find argmax of leaf probs for this tree
      int best_m = 0;
      double best_p = lp[0];
      for (int m = 1; m < M; ++m) {
        if (lp[m] > best_p) { best_p = lp[m]; best_m = m; }
      }
      vote_flat[static_cast<std::size_t>(i) * M + best_m] += 1;
    }
  }
#endif

  // -------------------------------------------------------------------------
  // Phase 3 — sequential: serialise native trees to R lists
  // -------------------------------------------------------------------------
  Rcpp::List forest_r(num_trees);
  for (int b = 0; b < num_trees; ++b)
    forest_r[b] = tree_to_list(native_forest[b], M);

  // Convert flat accumulator to (n x M) Rcpp matrix (row-major -> col-major)
  const double inv_B = 1.0 / static_cast<double>(num_trees);
  Rcpp::NumericMatrix predictions_r(n, M);
  for (int i = 0; i < n; ++i)
    for (int m = 0; m < M; ++m)
      predictions_r(i, m) = pred_flat[static_cast<std::size_t>(i) * M + m] * inv_B;

  // Convert vote accumulator to (n x M) Rcpp integer matrix
  Rcpp::IntegerMatrix votes_r(n, M);
  for (int i = 0; i < n; ++i)
    for (int m = 0; m < M; ++m)
      votes_r(i, m) = vote_flat[static_cast<std::size_t>(i) * M + m];

  return Rcpp::List::create(
    Rcpp::Named("predictions") = predictions_r,
    Rcpp::Named("forest")      = forest_r,
    Rcpp::Named("votes")       = votes_r
  );
}

// ===========================================================================
// grow_forest_oob_cpp (exported) — OOB error for hyperparameter tuning
// ===========================================================================

//' Grow a forest and compute debiased OOB error
//'
//' Variant of [grow_forest_cpp()] used by the built-in hyperparameter tuning
//' engine.  Instead of in-sample predictions and forest serialisation, this
//' function accumulates out-of-bag (OOB) predictions and returns a debiased
//' mean squared error scalar.  No forest or vote data is returned.
//'
//' @inheritParams grow_forest_cpp
//'
//' @return Named list: `oob_predictions` (n x M matrix, NaN where
//'   oob_count == 0), `debiased_error` (scalar).
//' @keywords internal
//' @export
// [[Rcpp::export]]
Rcpp::List grow_forest_oob_cpp(
  Rcpp::IntegerVector Y,
  Rcpp::NumericMatrix X,
  int                 num_trees,
  int                 min_node_size,
  int                 max_depth,
  int                 n_sub,
  int                 mtry,
  int                 M,
  Rcpp::NumericVector lambda,
  int                 num_threads = 0
) {
  const int n = Y.size();

  // Convert Y to 0-indexed plain int vector
  std::vector<int> y_vec(n);
  for (int i = 0; i < n; ++i) y_vec[i] = Y[i] - 1;

  const arma::mat x = Rcpp::as<arma::mat>(X);
  const int k       = static_cast<int>(x.n_cols);

  const double* lam_raw = lambda.begin();

  // Pre-transpose X to row-major for tree traversal
  const std::vector<double> x_rowmaj = to_rowmaj(x);

  // --- Phase 0: Global pre-sort ---
  SortData sort_data = build_sort_data(x);

  // --- Phase 1: Sequential R-RNG seeds + bootstrap ---
  Rcpp::NumericVector seed_doubles = Rcpp::runif(num_trees, 0.0, 4294967295.0);
  std::vector<uint32_t> tree_seeds(num_trees);
  for (int b = 0; b < num_trees; ++b)
    tree_seeds[b] = static_cast<uint32_t>(seed_doubles[b]);

  // Subsample without replacement; also build in_bag flags for OOB
  std::vector<std::vector<int>> boot_indices(num_trees);
  std::vector<std::vector<bool>> in_bag_flags(num_trees);
  for (int b = 0; b < num_trees; ++b) {
    Rcpp::IntegerVector boot_r = Rcpp::sample(n, n_sub, /*replace=*/false) - 1;
    boot_indices[b].assign(boot_r.begin(), boot_r.end());
    in_bag_flags[b].assign(n, false);
    for (int s = 0; s < n_sub; ++s)
      in_bag_flags[b][boot_indices[b][s]] = true;
  }

  // --- Phase 2: Parallel tree growing + OOB accumulation ---
  // OOB accumulators: sum of predictions, sum of squared predictions, count
  std::vector<double> oob_pred(static_cast<std::size_t>(n) * M, 0.0);
  std::vector<double> oob_pred_sq(static_cast<std::size_t>(n) * M, 0.0);
  std::vector<int>    oob_count(n, 0);

#ifdef _OPENMP
  const int nt_grow = (num_threads > 0) ? num_threads : omp_get_max_threads();
#pragma omp parallel num_threads(nt_grow)
  {
    std::vector<double> local_pred(static_cast<std::size_t>(n) * M, 0.0);
    std::vector<double> local_pred_sq(static_cast<std::size_t>(n) * M, 0.0);
    std::vector<int>    local_count(n, 0);

#pragma omp for schedule(dynamic, 4)
    for (int b = 0; b < num_trees; ++b) {
      std::mt19937 rng(tree_seeds[b]);
      TreeData td = grow_single_tree(
        y_vec.data(), sort_data,
        boot_indices[b].data(), n_sub,
        M, lam_raw, min_node_size, mtry, rng, max_depth
      );
      const double*      xr  = x_rowmaj.data();
      const std::vector<bool>& ib = in_bag_flags[b];

      for (int i = 0; i < n; ++i) {
        if (ib[i]) continue;  // skip in-bag observations
        const double* xi   = xr + static_cast<std::ptrdiff_t>(i) * k;
        const int     leaf = traverse_native(td, xi);
        const double* lp   = td.leaf_probs.data() + leaf * M;
        double*       acc  = local_pred.data() + static_cast<std::ptrdiff_t>(i) * M;
        double*       acc2 = local_pred_sq.data() + static_cast<std::ptrdiff_t>(i) * M;
        for (int m = 0; m < M; ++m) {
          acc[m]  += lp[m];
          acc2[m] += lp[m] * lp[m];
        }
        local_count[i] += 1;
      }
    }

#pragma omp critical
    {
      for (std::size_t idx = 0; idx < local_pred.size(); ++idx) {
        oob_pred[idx]    += local_pred[idx];
        oob_pred_sq[idx] += local_pred_sq[idx];
      }
      for (int i = 0; i < n; ++i)
        oob_count[i] += local_count[i];
    }
  }
#else
  for (int b = 0; b < num_trees; ++b) {
    std::mt19937 rng(tree_seeds[b]);
    TreeData td = grow_single_tree(
      y_vec.data(), sort_data,
      boot_indices[b].data(), n_sub,
      M, lam_raw, min_node_size, mtry, rng, max_depth
    );
    const double*      xr  = x_rowmaj.data();
    const std::vector<bool>& ib = in_bag_flags[b];

    for (int i = 0; i < n; ++i) {
      if (ib[i]) continue;
      const double* xi   = xr + static_cast<std::ptrdiff_t>(i) * k;
      const int     leaf = traverse_native(td, xi);
      const double* lp   = td.leaf_probs.data() + leaf * M;
      double*       acc  = oob_pred.data() + static_cast<std::ptrdiff_t>(i) * M;
      double*       acc2 = oob_pred_sq.data() + static_cast<std::ptrdiff_t>(i) * M;
      for (int m = 0; m < M; ++m) {
        acc[m]  += lp[m];
        acc2[m] += lp[m] * lp[m];
      }
      oob_count[i] += 1;
    }
  }
#endif

  // --- Compute OOB predictions and debiased error ---
  Rcpp::NumericMatrix oob_preds_r(n, M);
  double total_debiased = 0.0;
  int    n_valid = 0;

  for (int i = 0; i < n; ++i) {
    if (oob_count[i] < 2) {
      // Not enough OOB trees for variance estimate
      for (int m = 0; m < M; ++m) oob_preds_r(i, m) = R_NaN;
      continue;
    }
    const double cnt    = static_cast<double>(oob_count[i]);
    double raw_mse_i    = 0.0;
    double variance_i   = 0.0;

    for (int m = 0; m < M; ++m) {
      const std::size_t idx = static_cast<std::size_t>(i) * M + m;
      const double mean_p   = oob_pred[idx] / cnt;
      oob_preds_r(i, m)     = mean_p;
      const double indicator = (y_vec[i] == m) ? 1.0 : 0.0;
      raw_mse_i    += (indicator - mean_p) * (indicator - mean_p);
      variance_i   += (oob_pred_sq[idx] / cnt - mean_p * mean_p);
    }
    // Bessel correction for variance
    variance_i *= cnt / (cnt - 1.0);
    total_debiased += raw_mse_i - variance_i;
    ++n_valid;
  }

  const double debiased_error = (n_valid > 0)
    ? total_debiased / static_cast<double>(n_valid)
    : R_NaN;

  return Rcpp::List::create(
    Rcpp::Named("oob_predictions") = oob_preds_r,
    Rcpp::Named("debiased_error")  = debiased_error
  );
}

// ===========================================================================
// predict_forest_cpp (exported)
// ===========================================================================

//' Predict class probabilities from a fitted jocf forest
//'
//' Internal C++ engine called by [predict.jocf()].
//'
//' @param forest List of tree structures (from `grow_forest_cpp`).
//' @param X_new Numeric matrix of new observations (n_test x k).
//' @param M Number of outcome classes.
//' @param num_threads Number of OpenMP threads (0 = all available).
//'
//' @return Named list: `predictions` (n_test x M) and `votes` (n_test x M integer).
//' @keywords internal
//' @export
// [[Rcpp::export]]
Rcpp::List predict_forest_cpp(
  Rcpp::List          forest,
  Rcpp::NumericMatrix X_new,
  int                 M,
  int                 num_threads = 0
) {
  const int n_test    = X_new.nrow();
  const int num_trees = static_cast<int>(forest.size());

  const arma::mat x_new = Rcpp::as<arma::mat>(X_new);
  const int k           = static_cast<int>(x_new.n_cols);

  // Pre-transpose X_new to row-major
  const std::vector<double> x_rowmaj = to_rowmaj(x_new);

  // Deserialise all trees before entering the parallel region
  std::vector<TreeData> native_forest(num_trees);
  for (int b = 0; b < num_trees; ++b)
    native_forest[b] = list_to_tree(Rcpp::as<Rcpp::List>(forest[b]), M);

  std::vector<double> pred_flat(static_cast<std::size_t>(n_test) * M, 0.0);
  std::vector<int>    vote_flat(static_cast<std::size_t>(n_test) * M, 0);

#ifdef _OPENMP
  const int nt_pred = (num_threads > 0) ? num_threads : omp_get_max_threads();
#pragma omp parallel num_threads(nt_pred)
  {
    std::vector<double> local_pred(static_cast<std::size_t>(n_test) * M, 0.0);
    std::vector<int>    local_vote(static_cast<std::size_t>(n_test) * M, 0);

#pragma omp for schedule(dynamic)
    for (int b = 0; b < num_trees; ++b) {
      const TreeData& td  = native_forest[b];
      const double*   xr  = x_rowmaj.data();

      for (int i = 0; i < n_test; ++i) {
        const double* xi   = xr + static_cast<std::ptrdiff_t>(i) * k;
        const int     leaf = traverse_native(td, xi);
        const double* lp   = td.leaf_probs.data() + leaf * M;
        double*       acc  = local_pred.data() + static_cast<std::ptrdiff_t>(i) * M;
        for (int m = 0; m < M; ++m) acc[m] += lp[m];
        // Majority vote: find argmax of leaf probs for this tree
        int best_m = 0;
        double best_p = lp[0];
        for (int m = 1; m < M; ++m) {
          if (lp[m] > best_p) { best_p = lp[m]; best_m = m; }
        }
        local_vote[static_cast<std::size_t>(i) * M + best_m] += 1;
      }
    }

#pragma omp critical
    {
      for (std::size_t idx = 0; idx < local_pred.size(); ++idx)
        pred_flat[idx] += local_pred[idx];
      for (std::size_t idx = 0; idx < local_vote.size(); ++idx)
        vote_flat[idx] += local_vote[idx];
    }
  }
#else
  for (int b = 0; b < num_trees; ++b) {
    const TreeData& td  = native_forest[b];
    const double*   xr  = x_rowmaj.data();

    for (int i = 0; i < n_test; ++i) {
      const double* xi   = xr + static_cast<std::ptrdiff_t>(i) * k;
      const int     leaf = traverse_native(td, xi);
      const double* lp   = td.leaf_probs.data() + leaf * M;
      double*       acc  = pred_flat.data() + static_cast<std::ptrdiff_t>(i) * M;
      for (int m = 0; m < M; ++m) acc[m] += lp[m];
      // Majority vote: find argmax of leaf probs for this tree
      int best_m = 0;
      double best_p = lp[0];
      for (int m = 1; m < M; ++m) {
        if (lp[m] > best_p) { best_p = lp[m]; best_m = m; }
      }
      vote_flat[static_cast<std::size_t>(i) * M + best_m] += 1;
    }
  }
#endif

  const double inv_B = 1.0 / static_cast<double>(num_trees);
  Rcpp::NumericMatrix predictions_r(n_test, M);
  for (int i = 0; i < n_test; ++i)
    for (int m = 0; m < M; ++m)
      predictions_r(i, m) = pred_flat[static_cast<std::size_t>(i) * M + m] * inv_B;

  Rcpp::IntegerMatrix votes_r(n_test, M);
  for (int i = 0; i < n_test; ++i)
    for (int m = 0; m < M; ++m)
      votes_r(i, m) = vote_flat[static_cast<std::size_t>(i) * M + m];

  return Rcpp::List::create(
    Rcpp::Named("predictions") = predictions_r,
    Rcpp::Named("votes")       = votes_r
  );
}

// ===========================================================================
// marginal_effects_cpp (exported)
// ===========================================================================

//' Compute average marginal effects from a fitted jocf forest
//'
//' Internal C++ engine called by [marginal_effects.jocf()].
//'
//' @param forest List of tree structures (from `grow_forest_cpp`).
//' @param X_eval Numeric matrix of evaluation points (n_eval x k).
//' @param target_vars Integer vector of 0-based column indices to differentiate.
//' @param is_discrete Logical vector (length k_target); TRUE = discrete covariate.
//' @param h_vec Numeric vector (length k_target); step size omega*sd_j for
//'   continuous variables (ignored for discrete).
//' @param M Number of outcome classes.
//' @param num_threads Number of OpenMP threads (0 = all available).
//'
//' @return Numeric matrix (k_target x M) of average marginal effects.
//' @keywords internal
//' @export
// [[Rcpp::export]]
Rcpp::NumericMatrix marginal_effects_cpp(
  Rcpp::List          forest,
  Rcpp::NumericMatrix X_eval,
  Rcpp::IntegerVector target_vars,
  Rcpp::LogicalVector is_discrete,
  Rcpp::NumericVector h_vec,
  int                 M,
  int                 num_threads = 0
) {
  const int num_trees = static_cast<int>(forest.size());
  const int n_eval    = X_eval.nrow();
  const int k         = X_eval.ncol();
  const int k_target  = static_cast<int>(target_vars.size());

  const arma::mat x_eval_arma = Rcpp::as<arma::mat>(X_eval);

  // Pre-transpose X_eval to row-major; eliminates arma element access in loop
  const std::vector<double> x_rowmaj = to_rowmaj(x_eval_arma);

  // Deserialise all trees before the parallel region
  std::vector<TreeData> native_forest(num_trees);
  for (int b = 0; b < num_trees; ++b)
    native_forest[b] = list_to_tree(Rcpp::as<Rcpp::List>(forest[b]), M);

  // Copy target metadata to plain C++ containers (safe inside OpenMP)
  std::vector<int>    t_col(k_target);
  std::vector<bool>   t_disc(k_target);
  std::vector<double> t_h(k_target);
  for (int jt = 0; jt < k_target; ++jt) {
    t_col[jt]  = target_vars[jt];   // 0-based
    t_disc[jt] = static_cast<bool>(is_discrete[jt]);
    t_h[jt]    = h_vec[jt];
  }

  // Global accumulator: flat [jt * M + m]
  std::vector<double> effects_acc(static_cast<std::size_t>(k_target) * M, 0.0);

#ifdef _OPENMP
  const int nt_me = (num_threads > 0) ? num_threads : omp_get_max_threads();
#pragma omp parallel num_threads(nt_me)
  {
    std::vector<double> local_eff(static_cast<std::size_t>(k_target) * M, 0.0);
    std::vector<double> xi(k);

#pragma omp for schedule(static)
    for (int b = 0; b < num_trees; ++b) {
      const TreeData& td  = native_forest[b];
      const double*   xr  = x_rowmaj.data();

      for (int i = 0; i < n_eval; ++i) {
        // Copy row i from row-major layout (contiguous read)
        const double* xi_src = xr + static_cast<std::ptrdiff_t>(i) * k;
        std::copy(xi_src, xi_src + k, xi.begin());

        for (int jt = 0; jt < k_target; ++jt) {
          const int    col    = t_col[jt];
          const double x_orig = xi[col];
          int    leaf_hi, leaf_lo;
          double scale;

          if (t_disc[jt]) {
            xi[col]  = std::floor(x_orig);
            leaf_lo  = traverse_native(td, xi);
            xi[col]  = std::floor(x_orig) + 1.0;
            leaf_hi  = traverse_native(td, xi);
            scale    = 1.0;
          } else {
            xi[col]  = x_orig - t_h[jt];
            leaf_lo  = traverse_native(td, xi);
            xi[col]  = x_orig + t_h[jt];
            leaf_hi  = traverse_native(td, xi);
            scale    = 1.0 / (2.0 * t_h[jt]);
          }
          xi[col] = x_orig;  // restore

          const int base_hi = leaf_hi * M;
          const int base_lo = leaf_lo * M;
          double* out = local_eff.data() + static_cast<std::ptrdiff_t>(jt) * M;
          const double* lp_hi = td.leaf_probs.data() + base_hi;
          const double* lp_lo = td.leaf_probs.data() + base_lo;
          for (int m = 0; m < M; ++m)
            out[m] += (lp_hi[m] - lp_lo[m]) * scale;
        }
      }
    }

#pragma omp critical
    {
      for (std::size_t i = 0; i < local_eff.size(); ++i)
        effects_acc[i] += local_eff[i];
    }
  }
#else
  {
    std::vector<double> xi(k);
    const double* xr = x_rowmaj.data();

    for (int b = 0; b < num_trees; ++b) {
      const TreeData& td = native_forest[b];

      for (int i = 0; i < n_eval; ++i) {
        const double* xi_src = xr + static_cast<std::ptrdiff_t>(i) * k;
        std::copy(xi_src, xi_src + k, xi.begin());

        for (int jt = 0; jt < k_target; ++jt) {
          const int    col    = t_col[jt];
          const double x_orig = xi[col];
          int    leaf_hi, leaf_lo;
          double scale;

          if (t_disc[jt]) {
            xi[col]  = std::floor(x_orig);
            leaf_lo  = traverse_native(td, xi);
            xi[col]  = std::floor(x_orig) + 1.0;
            leaf_hi  = traverse_native(td, xi);
            scale    = 1.0;
          } else {
            xi[col]  = x_orig - t_h[jt];
            leaf_lo  = traverse_native(td, xi);
            xi[col]  = x_orig + t_h[jt];
            leaf_hi  = traverse_native(td, xi);
            scale    = 1.0 / (2.0 * t_h[jt]);
          }
          xi[col] = x_orig;  // restore

          const int base_hi = leaf_hi * M;
          const int base_lo = leaf_lo * M;
          double* out = effects_acc.data() + static_cast<std::ptrdiff_t>(jt) * M;
          const double* lp_hi = td.leaf_probs.data() + base_hi;
          const double* lp_lo = td.leaf_probs.data() + base_lo;
          for (int m = 0; m < M; ++m)
            out[m] += (lp_hi[m] - lp_lo[m]) * scale;
        }
      }
    }
  }
#endif

  const double denom = static_cast<double>(num_trees) * static_cast<double>(n_eval);
  for (std::size_t i = 0; i < effects_acc.size(); ++i)
    effects_acc[i] /= denom;

  // Return (k_target x M) matrix
  Rcpp::NumericMatrix result(k_target, M);
  for (int jt = 0; jt < k_target; ++jt)
    for (int m = 0; m < M; ++m)
      result(jt, m) = effects_acc[static_cast<std::size_t>(jt) * M + m];

  return result;
}

// ===========================================================================
// Honest forest support: HonestTree, serialisation, grow, predict, ME
// ===========================================================================

// ---------------------------------------------------------------------------
// HonestTree: TreeData + sorted-by-leaf honesty data
// ---------------------------------------------------------------------------
struct HonestTree {
  TreeData td;
  std::vector<int> hon_sorted;   // hon obs indices sorted by leaf
  std::vector<int> hon_offsets;  // boundary array: hon obs in leaf L are
                                  // hon_sorted[hon_offsets[L]..hon_offsets[L+1])
};

// ---------------------------------------------------------------------------
// tree_to_list_honest: serialise HonestTree to R list
// ---------------------------------------------------------------------------
static Rcpp::List tree_to_list_honest(const HonestTree& ht, int M) {
  Rcpp::List base = tree_to_list(ht.td, M);
  base["hon_sorted"]  = Rcpp::IntegerVector(ht.hon_sorted.begin(),
                                             ht.hon_sorted.end());
  base["hon_offsets"] = Rcpp::IntegerVector(ht.hon_offsets.begin(),
                                             ht.hon_offsets.end());
  return base;
}

// ---------------------------------------------------------------------------
// list_to_honest_tree: deserialise R list to HonestTree
// ---------------------------------------------------------------------------
static HonestTree list_to_honest_tree(const Rcpp::List& tree, int M) {
  HonestTree ht;
  ht.td = list_to_tree(tree, M);
  Rcpp::IntegerVector hs = tree["hon_sorted"];
  Rcpp::IntegerVector ho = tree["hon_offsets"];
  ht.hon_sorted.assign(hs.begin(), hs.end());
  ht.hon_offsets.assign(ho.begin(), ho.end());
  return ht;
}

// ===========================================================================
// grow_forest_honest_cpp (exported)
// ===========================================================================

//' Grow an honest joint OCF random forest
//'
//' Internal C++ engine for honest forests.  Trees are grown on S^tr and
//' repopulated with S^hon for leaf predictions.  Each tree stores sorted-by-
//' leaf honesty data for weight-based inference at predict time.
//'
//' @param Y Integer vector of class labels (1..M), length n.
//' @param X Numeric matrix of covariates (n x k).
//' @param num_trees Number of trees.
//' @param min_node_size Minimum observations per terminal node.
//' @param max_depth Maximum tree depth (-1 = unlimited).
//' @param n_sub_tr Subsample size drawn from S^tr (without replacement).
//' @param mtry Number of candidate features at each split.
//' @param M Number of outcome classes.
//' @param lambda Numeric weight vector of length M.
//' @param tr_indices Integer vector of 0-based training indices.
//' @param hon_indices Integer vector of 0-based honesty indices.
//' @param num_threads Number of OpenMP threads (0 = all available).
//'
//' @return Named list: `predictions` (n x M), `forest` (list of B trees
//'   with hon_sorted/hon_offsets), `votes` (n x M integer), `n_hon` (int).
//' @keywords internal
//' @export
// [[Rcpp::export]]
Rcpp::List grow_forest_honest_cpp(
  Rcpp::IntegerVector Y,
  Rcpp::NumericMatrix X,
  int                 num_trees,
  int                 min_node_size,
  int                 max_depth,
  int                 n_sub_tr,
  int                 mtry,
  int                 M,
  Rcpp::NumericVector lambda,
  Rcpp::IntegerVector tr_indices,
  Rcpp::IntegerVector hon_indices,
  int                 num_threads = 0
) {
  const int n     = Y.size();
  const int n_tr  = tr_indices.size();
  const int n_hon = hon_indices.size();

  // Convert Y to 0-indexed
  std::vector<int> y_vec(n);
  for (int i = 0; i < n; ++i) y_vec[i] = Y[i] - 1;

  const arma::mat x = Rcpp::as<arma::mat>(X);
  const int k       = static_cast<int>(x.n_cols);
  const double* lam_raw = lambda.begin();

  // Copy indices to plain vectors
  std::vector<int> tr_idx(tr_indices.begin(), tr_indices.end());
  std::vector<int> hon_idx(hon_indices.begin(), hon_indices.end());

  // Pre-transpose X to row-major for prediction loops
  const std::vector<double> x_rowmaj = to_rowmaj(x);

  // Phase 0 — Global pre-sort over all n observations
  SortData sort_data = build_sort_data(x);

  // Phase 1 — Sequential: seeds + bootstrap from tr_indices only
  Rcpp::NumericVector seed_doubles = Rcpp::runif(num_trees, 0.0, 4294967295.0);
  std::vector<uint32_t> tree_seeds(num_trees);
  for (int b = 0; b < num_trees; ++b)
    tree_seeds[b] = static_cast<uint32_t>(seed_doubles[b]);

  std::vector<std::vector<int>> boot_indices(num_trees);
  for (int b = 0; b < num_trees; ++b) {
    Rcpp::IntegerVector draw = Rcpp::sample(n_tr, n_sub_tr, /*replace=*/false) - 1;
    boot_indices[b].resize(n_sub_tr);
    for (int s = 0; s < n_sub_tr; ++s)
      boot_indices[b][s] = tr_idx[draw[s]];
  }

  // Phase 2 — Parallel: grow trees + repopulate with S^hon
  std::vector<HonestTree> honest_forest(num_trees);
  std::vector<double> pred_flat(static_cast<std::size_t>(n) * M, 0.0);
  std::vector<int>    vote_flat(static_cast<std::size_t>(n) * M, 0);

#ifdef _OPENMP
  const int nt_grow = (num_threads > 0) ? num_threads : omp_get_max_threads();
#pragma omp parallel num_threads(nt_grow)
  {
    std::vector<double> local_pred(static_cast<std::size_t>(n) * M, 0.0);
    std::vector<int>    local_vote(static_cast<std::size_t>(n) * M, 0);
    // Per-thread scratch for repopulation
    std::vector<int> leaf_assignment(n_hon);
    std::vector<int> node_count_buf;

#pragma omp for schedule(dynamic, 4)
    for (int b = 0; b < num_trees; ++b) {
      std::mt19937 rng(tree_seeds[b]);
      TreeData td = grow_single_tree(
        y_vec.data(), sort_data,
        boot_indices[b].data(), n_sub_tr,
        M, lam_raw, min_node_size, mtry, rng, max_depth
      );
      const int n_nodes = td.n_nodes;

      // --- Repopulate: traverse all hon obs through the tree ---
      const double* xr = x_rowmaj.data();
      for (int h = 0; h < n_hon; ++h) {
        const int obs = hon_idx[h];
        const double* xi = xr + static_cast<std::ptrdiff_t>(obs) * k;
        leaf_assignment[h] = traverse_native(td, xi);
      }

      // Count hon obs per node
      node_count_buf.assign(n_nodes, 0);
      for (int h = 0; h < n_hon; ++h)
        node_count_buf[leaf_assignment[h]]++;

      // Prefix sum → offsets
      std::vector<int> offsets(n_nodes + 1, 0);
      for (int nd = 0; nd < n_nodes; ++nd)
        offsets[nd + 1] = offsets[nd] + node_count_buf[nd];

      // Scatter into sorted array
      std::vector<int> hon_sorted(n_hon);
      std::vector<int> write_pos(offsets.begin(), offsets.begin() + n_nodes);
      for (int h = 0; h < n_hon; ++h) {
        const int nd = leaf_assignment[h];
        hon_sorted[write_pos[nd]++] = h;  // store position in hon_idx (0..n_hon-1)
      }

      // Recompute leaf_probs from hon counts only
      for (int nd = 0; nd < n_nodes; ++nd) {
        const int cnt = offsets[nd + 1] - offsets[nd];
        if (cnt == 0) {
          // Empty leaf: uniform
          for (int m = 0; m < M; ++m)
            td.leaf_probs[nd * M + m] = 1.0 / static_cast<double>(M);
        } else {
          // Count classes among hon obs in this node
          for (int m = 0; m < M; ++m)
            td.leaf_probs[nd * M + m] = 0.0;
          for (int pos = offsets[nd]; pos < offsets[nd + 1]; ++pos) {
            const int h_idx = hon_sorted[pos];
            const int cls = y_vec[hon_idx[h_idx]];
            td.leaf_probs[nd * M + cls] += 1.0;
          }
          const double inv_cnt = 1.0 / static_cast<double>(cnt);
          for (int m = 0; m < M; ++m)
            td.leaf_probs[nd * M + m] *= inv_cnt;
        }
      }

      // Store honest tree
      HonestTree& ht = honest_forest[b];
      ht.td = std::move(td);
      ht.hon_sorted = std::move(hon_sorted);
      ht.hon_offsets = std::move(offsets);

      // Accumulate in-sample predictions + votes (all n obs through honest tree)
      for (int i = 0; i < n; ++i) {
        const double* xi = xr + static_cast<std::ptrdiff_t>(i) * k;
        const int leaf = traverse_native(ht.td, xi);
        const double* lp = ht.td.leaf_probs.data() + leaf * M;
        double* acc = local_pred.data() + static_cast<std::ptrdiff_t>(i) * M;
        for (int m = 0; m < M; ++m) acc[m] += lp[m];
        int best_m = 0;
        double best_p = lp[0];
        for (int m = 1; m < M; ++m) {
          if (lp[m] > best_p) { best_p = lp[m]; best_m = m; }
        }
        local_vote[static_cast<std::size_t>(i) * M + best_m] += 1;
      }
    }

#pragma omp critical
    {
      for (std::size_t idx = 0; idx < local_pred.size(); ++idx)
        pred_flat[idx] += local_pred[idx];
      for (std::size_t idx = 0; idx < local_vote.size(); ++idx)
        vote_flat[idx] += local_vote[idx];
    }
  }
#else
  {
    std::vector<int> leaf_assignment(n_hon);
    std::vector<int> node_count_buf;

    for (int b = 0; b < num_trees; ++b) {
      std::mt19937 rng(tree_seeds[b]);
      TreeData td = grow_single_tree(
        y_vec.data(), sort_data,
        boot_indices[b].data(), n_sub_tr,
        M, lam_raw, min_node_size, mtry, rng, max_depth
      );
      const int n_nodes = td.n_nodes;

      const double* xr = x_rowmaj.data();
      for (int h = 0; h < n_hon; ++h) {
        const int obs = hon_idx[h];
        const double* xi = xr + static_cast<std::ptrdiff_t>(obs) * k;
        leaf_assignment[h] = traverse_native(td, xi);
      }

      node_count_buf.assign(n_nodes, 0);
      for (int h = 0; h < n_hon; ++h)
        node_count_buf[leaf_assignment[h]]++;

      std::vector<int> offsets(n_nodes + 1, 0);
      for (int nd = 0; nd < n_nodes; ++nd)
        offsets[nd + 1] = offsets[nd] + node_count_buf[nd];

      std::vector<int> hon_sorted(n_hon);
      std::vector<int> write_pos(offsets.begin(), offsets.begin() + n_nodes);
      for (int h = 0; h < n_hon; ++h) {
        const int nd = leaf_assignment[h];
        hon_sorted[write_pos[nd]++] = h;
      }

      for (int nd = 0; nd < n_nodes; ++nd) {
        const int cnt = offsets[nd + 1] - offsets[nd];
        if (cnt == 0) {
          for (int m = 0; m < M; ++m)
            td.leaf_probs[nd * M + m] = 1.0 / static_cast<double>(M);
        } else {
          for (int m = 0; m < M; ++m)
            td.leaf_probs[nd * M + m] = 0.0;
          for (int pos = offsets[nd]; pos < offsets[nd + 1]; ++pos) {
            const int h_idx = hon_sorted[pos];
            const int cls = y_vec[hon_idx[h_idx]];
            td.leaf_probs[nd * M + cls] += 1.0;
          }
          const double inv_cnt = 1.0 / static_cast<double>(cnt);
          for (int m = 0; m < M; ++m)
            td.leaf_probs[nd * M + m] *= inv_cnt;
        }
      }

      HonestTree& ht = honest_forest[b];
      ht.td = std::move(td);
      ht.hon_sorted = std::move(hon_sorted);
      ht.hon_offsets = std::move(offsets);

      for (int i = 0; i < n; ++i) {
        const double* xi = xr + static_cast<std::ptrdiff_t>(i) * k;
        const int leaf = traverse_native(ht.td, xi);
        const double* lp = ht.td.leaf_probs.data() + leaf * M;
        double* acc = pred_flat.data() + static_cast<std::ptrdiff_t>(i) * M;
        for (int m = 0; m < M; ++m) acc[m] += lp[m];
        int best_m = 0;
        double best_p = lp[0];
        for (int m = 1; m < M; ++m) {
          if (lp[m] > best_p) { best_p = lp[m]; best_m = m; }
        }
        vote_flat[static_cast<std::size_t>(i) * M + best_m] += 1;
      }
    }
  }
#endif

  // Phase 3 — Serialize
  Rcpp::List forest_r(num_trees);
  for (int b = 0; b < num_trees; ++b)
    forest_r[b] = tree_to_list_honest(honest_forest[b], M);

  const double inv_B = 1.0 / static_cast<double>(num_trees);
  Rcpp::NumericMatrix predictions_r(n, M);
  for (int i = 0; i < n; ++i)
    for (int m = 0; m < M; ++m)
      predictions_r(i, m) = pred_flat[static_cast<std::size_t>(i) * M + m] * inv_B;

  Rcpp::IntegerMatrix votes_r(n, M);
  for (int i = 0; i < n; ++i)
    for (int m = 0; m < M; ++m)
      votes_r(i, m) = vote_flat[static_cast<std::size_t>(i) * M + m];

  return Rcpp::List::create(
    Rcpp::Named("predictions") = predictions_r,
    Rcpp::Named("forest")      = forest_r,
    Rcpp::Named("votes")       = votes_r,
    Rcpp::Named("n_hon")       = n_hon
  );
}

// ===========================================================================
// predict_honest_cpp (exported)
// ===========================================================================

//' Predict with variance estimation from an honest jocf forest
//'
//' Weight-based variance estimation: for each test point, computes alpha weights
//' from honest leaf memberships across all trees, then derives variance from
//' the weighted sum of squared indicator deviations.
//'
//' @param forest List of honest tree structures.
//' @param X_new Numeric matrix of new observations (n_test x k).
//' @param Y_hon Integer vector of 0-indexed class labels for honesty sample.
//' @param n_hon Number of honesty observations.
//' @param M Number of outcome classes.
//' @param compute_variance Logical; if TRUE compute variance estimates.
//' @param num_threads Number of OpenMP threads (0 = all available).
//'
//' @return Named list: `predictions` (n_test x M), `variance` (n_test x M or
//'   empty), `votes` (n_test x M integer).
//' @keywords internal
//' @export
// [[Rcpp::export]]
Rcpp::List predict_honest_cpp(
  Rcpp::List          forest,
  Rcpp::NumericMatrix X_new,
  Rcpp::IntegerVector Y_hon,
  int                 n_hon,
  int                 M,
  bool                compute_variance,
  int                 num_threads = 0
) {
  const int n_test    = X_new.nrow();
  const int num_trees = static_cast<int>(forest.size());

  const arma::mat x_new = Rcpp::as<arma::mat>(X_new);
  const int k           = static_cast<int>(x_new.n_cols);

  const std::vector<double> x_rowmaj = to_rowmaj(x_new);

  // Copy Y_hon to plain vector
  std::vector<int> y_hon_vec(Y_hon.begin(), Y_hon.end());

  // Deserialise all trees before parallel region
  std::vector<HonestTree> htrees(num_trees);
  for (int b = 0; b < num_trees; ++b)
    htrees[b] = list_to_honest_tree(Rcpp::as<Rcpp::List>(forest[b]), M);

  if (!compute_variance) {
    // Simple prediction: just use leaf_probs (already honest)
    // Parallelize over trees like predict_forest_cpp
    std::vector<double> pred_flat(static_cast<std::size_t>(n_test) * M, 0.0);
    std::vector<int>    vote_flat(static_cast<std::size_t>(n_test) * M, 0);

#ifdef _OPENMP
    const int nt = (num_threads > 0) ? num_threads : omp_get_max_threads();
#pragma omp parallel num_threads(nt)
    {
      std::vector<double> lp(static_cast<std::size_t>(n_test) * M, 0.0);
      std::vector<int>    lv(static_cast<std::size_t>(n_test) * M, 0);
#pragma omp for schedule(dynamic)
      for (int b = 0; b < num_trees; ++b) {
        const TreeData& td = htrees[b].td;
        for (int i = 0; i < n_test; ++i) {
          const double* xi = x_rowmaj.data()
                             + static_cast<std::ptrdiff_t>(i) * k;
          const int leaf = traverse_native(td, xi);
          const double* p = td.leaf_probs.data() + leaf * M;
          double* acc = lp.data() + static_cast<std::ptrdiff_t>(i) * M;
          for (int m = 0; m < M; ++m) acc[m] += p[m];
          int bm = 0; double bp = p[0];
          for (int m = 1; m < M; ++m)
            if (p[m] > bp) { bp = p[m]; bm = m; }
          lv[static_cast<std::size_t>(i) * M + bm] += 1;
        }
      }
#pragma omp critical
      {
        for (std::size_t idx = 0; idx < lp.size(); ++idx) pred_flat[idx] += lp[idx];
        for (std::size_t idx = 0; idx < lv.size(); ++idx) vote_flat[idx] += lv[idx];
      }
    }
#else
    for (int b = 0; b < num_trees; ++b) {
      const TreeData& td = htrees[b].td;
      for (int i = 0; i < n_test; ++i) {
        const double* xi = x_rowmaj.data()
                           + static_cast<std::ptrdiff_t>(i) * k;
        const int leaf = traverse_native(td, xi);
        const double* p = td.leaf_probs.data() + leaf * M;
        double* acc = pred_flat.data() + static_cast<std::ptrdiff_t>(i) * M;
        for (int m = 0; m < M; ++m) acc[m] += p[m];
        int bm = 0; double bp = p[0];
        for (int m = 1; m < M; ++m)
          if (p[m] > bp) { bp = p[m]; bm = m; }
        vote_flat[static_cast<std::size_t>(i) * M + bm] += 1;
      }
    }
#endif

    const double inv_B = 1.0 / static_cast<double>(num_trees);
    Rcpp::NumericMatrix preds(n_test, M);
    for (int i = 0; i < n_test; ++i)
      for (int m = 0; m < M; ++m)
        preds(i, m) = pred_flat[static_cast<std::size_t>(i) * M + m] * inv_B;

    Rcpp::IntegerMatrix votes(n_test, M);
    for (int i = 0; i < n_test; ++i)
      for (int m = 0; m < M; ++m)
        votes(i, m) = vote_flat[static_cast<std::size_t>(i) * M + m];

    return Rcpp::List::create(
      Rcpp::Named("predictions") = preds,
      Rcpp::Named("variance")    = Rcpp::NumericMatrix(0, 0),
      Rcpp::Named("votes")       = votes
    );
  }

  // --- Variance computation: parallelize over test points ---
  Rcpp::NumericMatrix preds(n_test, M);
  Rcpp::NumericMatrix var_mat(n_test, M);
  std::vector<int> vote_flat(static_cast<std::size_t>(n_test) * M, 0);

#ifdef _OPENMP
  const int nt = (num_threads > 0) ? num_threads : omp_get_max_threads();
#pragma omp parallel num_threads(nt)
  {
    std::vector<double> omega(n_hon, 0.0);
    std::vector<int> local_vote(static_cast<std::size_t>(n_test) * M, 0);

#pragma omp for schedule(dynamic)
    for (int i = 0; i < n_test; ++i) {
      const double* xi = x_rowmaj.data() + static_cast<std::ptrdiff_t>(i) * k;

      // Zero omega
      std::memset(omega.data(), 0, static_cast<std::size_t>(n_hon) * sizeof(double));

      // Accumulate omega weights across all trees
      for (int b = 0; b < num_trees; ++b) {
        const HonestTree& ht = htrees[b];
        const int leaf = traverse_native(ht.td, xi);
        const int start = ht.hon_offsets[leaf];
        const int end   = ht.hon_offsets[leaf + 1];
        const int leaf_size = end - start;

        // Vote counting
        const double* lp = ht.td.leaf_probs.data() + leaf * M;
        int bm = 0; double bp = lp[0];
        for (int m = 1; m < M; ++m)
          if (lp[m] > bp) { bp = lp[m]; bm = m; }
        local_vote[static_cast<std::size_t>(i) * M + bm] += 1;

        if (leaf_size > 0) {
          const double w = 1.0 / (static_cast<double>(num_trees)
                                  * static_cast<double>(leaf_size));
          for (int pos = start; pos < end; ++pos)
            omega[ht.hon_sorted[pos]] += w;
        }
      }

      // Compute prediction and variance for each class
      for (int m = 0; m < M; ++m) {
        double sum_z  = 0.0;
        double sum_z2 = 0.0;
        for (int h = 0; h < n_hon; ++h) {
          if (omega[h] == 0.0) continue;
          const double z = (y_hon_vec[h] == m) ? 1.0 : 0.0;
          const double wz = omega[h] * z;
          sum_z  += wz;
          sum_z2 += wz * wz;
        }
        preds(i, m) = sum_z;
        // Variance: V = n_hon / (n_hon - 1) * (Σ (ω_h z_h)² - (Σ ω_h z_h)²/n_hon)
        // Simplified: (n_hon * Σ(ωz)² - (Σωz)²) / (n_hon - 1)
        if (n_hon > 1) {
          const double v = (static_cast<double>(n_hon) * sum_z2 - sum_z * sum_z)
                           / static_cast<double>(n_hon - 1);
          var_mat(i, m) = std::max(v, 0.0);
        } else {
          var_mat(i, m) = 0.0;
        }
      }
    }

#pragma omp critical
    {
      for (std::size_t idx = 0; idx < local_vote.size(); ++idx)
        vote_flat[idx] += local_vote[idx];
    }
  }
#else
  {
    std::vector<double> omega(n_hon, 0.0);
    for (int i = 0; i < n_test; ++i) {
      const double* xi = x_rowmaj.data() + static_cast<std::ptrdiff_t>(i) * k;
      std::memset(omega.data(), 0, static_cast<std::size_t>(n_hon) * sizeof(double));

      for (int b = 0; b < num_trees; ++b) {
        const HonestTree& ht = htrees[b];
        const int leaf = traverse_native(ht.td, xi);
        const int start = ht.hon_offsets[leaf];
        const int end   = ht.hon_offsets[leaf + 1];
        const int leaf_size = end - start;

        const double* lp = ht.td.leaf_probs.data() + leaf * M;
        int bm = 0; double bp = lp[0];
        for (int m = 1; m < M; ++m)
          if (lp[m] > bp) { bp = lp[m]; bm = m; }
        vote_flat[static_cast<std::size_t>(i) * M + bm] += 1;

        if (leaf_size > 0) {
          const double w = 1.0 / (static_cast<double>(num_trees)
                                  * static_cast<double>(leaf_size));
          for (int pos = start; pos < end; ++pos)
            omega[ht.hon_sorted[pos]] += w;
        }
      }

      for (int m = 0; m < M; ++m) {
        double sum_z  = 0.0;
        double sum_z2 = 0.0;
        for (int h = 0; h < n_hon; ++h) {
          if (omega[h] == 0.0) continue;
          const double z = (y_hon_vec[h] == m) ? 1.0 : 0.0;
          const double wz = omega[h] * z;
          sum_z  += wz;
          sum_z2 += wz * wz;
        }
        preds(i, m) = sum_z;
        if (n_hon > 1) {
          const double v = (static_cast<double>(n_hon) * sum_z2 - sum_z * sum_z)
                           / static_cast<double>(n_hon - 1);
          var_mat(i, m) = std::max(v, 0.0);
        } else {
          var_mat(i, m) = 0.0;
        }
      }
    }
  }
#endif

  Rcpp::IntegerMatrix votes(n_test, M);
  for (int i = 0; i < n_test; ++i)
    for (int m = 0; m < M; ++m)
      votes(i, m) = vote_flat[static_cast<std::size_t>(i) * M + m];

  return Rcpp::List::create(
    Rcpp::Named("predictions") = preds,
    Rcpp::Named("variance")    = var_mat,
    Rcpp::Named("votes")       = votes
  );
}

// ===========================================================================
// marginal_effects_honest_cpp (exported)
// ===========================================================================

//' Compute marginal effects with standard errors from an honest jocf forest
//'
//' Weight-based SE estimation for average marginal effects.  For each eval
//' point, computes omega_plus and omega_minus weights from perturbed
//' traversals, then derives pointwise variance from the weighted indicator
//' differences.
//'
//' @param forest List of honest tree structures.
//' @param X_eval Numeric matrix of evaluation points (n_eval x k).
//' @param target_vars Integer vector of 0-based column indices.
//' @param is_discrete Logical vector; TRUE = discrete covariate.
//' @param h_vec Numeric vector of step sizes for continuous variables.
//' @param Y_hon Integer vector of 0-indexed class labels for honesty sample.
//' @param n_hon Number of honesty observations.
//' @param M Number of outcome classes.
//' @param num_threads Number of OpenMP threads (0 = all available).
//'
//' @return Named list: `AME` (k_target x M), `variance` (k_target x M).
//' @keywords internal
//' @export
// [[Rcpp::export]]
Rcpp::List marginal_effects_honest_cpp(
  Rcpp::List          forest,
  Rcpp::NumericMatrix X_eval,
  Rcpp::IntegerVector target_vars,
  Rcpp::LogicalVector is_discrete,
  Rcpp::NumericVector h_vec,
  Rcpp::IntegerVector Y_hon,
  int                 n_hon,
  int                 M,
  int                 num_threads = 0
) {
  const int num_trees = static_cast<int>(forest.size());
  const int n_eval    = X_eval.nrow();
  const int k         = X_eval.ncol();
  const int k_target  = static_cast<int>(target_vars.size());

  const arma::mat x_eval_arma = Rcpp::as<arma::mat>(X_eval);
  const std::vector<double> x_rowmaj = to_rowmaj(x_eval_arma);

  // Copy to plain vectors
  std::vector<int> y_hon_vec(Y_hon.begin(), Y_hon.end());
  std::vector<int>    t_col(k_target);
  std::vector<bool>   t_disc(k_target);
  std::vector<double> t_h(k_target);
  for (int jt = 0; jt < k_target; ++jt) {
    t_col[jt]  = target_vars[jt];
    t_disc[jt] = static_cast<bool>(is_discrete[jt]);
    t_h[jt]    = h_vec[jt];
  }

  // Deserialise trees
  std::vector<HonestTree> htrees(num_trees);
  for (int b = 0; b < num_trees; ++b)
    htrees[b] = list_to_honest_tree(Rcpp::as<Rcpp::List>(forest[b]), M);

  const double inv_B = 1.0 / static_cast<double>(num_trees);

  // Global accumulators: flat [jt * M + m]
  const std::size_t km = static_cast<std::size_t>(k_target) * M;
  std::vector<double> ame_acc(km, 0.0);
  std::vector<double> var_acc(km, 0.0);

#ifdef _OPENMP
  const int nt = (num_threads > 0) ? num_threads : omp_get_max_threads();
#pragma omp parallel num_threads(nt)
  {
    std::vector<double> local_ame(km, 0.0);
    std::vector<double> local_var(km, 0.0);
    std::vector<double> omega_plus(n_hon);
    std::vector<double> omega_minus(n_hon);
    std::vector<double> xi(k);

#pragma omp for schedule(dynamic)
    for (int i = 0; i < n_eval; ++i) {
      const double* xi_src = x_rowmaj.data() + static_cast<std::ptrdiff_t>(i) * k;

      for (int jt = 0; jt < k_target; ++jt) {
        std::copy(xi_src, xi_src + k, xi.begin());
        const int col = t_col[jt];
        const double x_orig = xi[col];
        double scale;

        // Compute x_plus and x_minus
        double x_plus_val, x_minus_val;
        if (t_disc[jt]) {
          x_minus_val = std::floor(x_orig);
          x_plus_val  = std::floor(x_orig) + 1.0;
          scale = 1.0;
        } else {
          x_minus_val = x_orig - t_h[jt];
          x_plus_val  = x_orig + t_h[jt];
          scale = 1.0 / (2.0 * t_h[jt]);
        }

        // Accumulate omega_plus and omega_minus across all trees
        std::memset(omega_plus.data(), 0, n_hon * sizeof(double));
        std::memset(omega_minus.data(), 0, n_hon * sizeof(double));

        for (int b = 0; b < num_trees; ++b) {
          const HonestTree& ht = htrees[b];

          xi[col] = x_plus_val;
          const int leaf_plus = traverse_native(ht.td, xi);
          const int sp = ht.hon_offsets[leaf_plus];
          const int ep = ht.hon_offsets[leaf_plus + 1];
          const int lsz_p = ep - sp;
          if (lsz_p > 0) {
            const double w = inv_B / static_cast<double>(lsz_p);
            for (int pos = sp; pos < ep; ++pos)
              omega_plus[ht.hon_sorted[pos]] += w;
          }

          xi[col] = x_minus_val;
          const int leaf_minus = traverse_native(ht.td, xi);
          const int sm = ht.hon_offsets[leaf_minus];
          const int em = ht.hon_offsets[leaf_minus + 1];
          const int lsz_m = em - sm;
          if (lsz_m > 0) {
            const double w = inv_B / static_cast<double>(lsz_m);
            for (int pos = sm; pos < em; ++pos)
              omega_minus[ht.hon_sorted[pos]] += w;
          }
        }
        xi[col] = x_orig;

        // Compute effect and variance for each class
        double* ame_out = local_ame.data() + static_cast<std::ptrdiff_t>(jt) * M;
        double* var_out = local_var.data() + static_cast<std::ptrdiff_t>(jt) * M;
        for (int m = 0; m < M; ++m) {
          double sum_dz  = 0.0;
          double sum_dz2 = 0.0;
          for (int h = 0; h < n_hon; ++h) {
            const double dw = omega_plus[h] - omega_minus[h];
            if (dw == 0.0) continue;
            const double z = (y_hon_vec[h] == m) ? 1.0 : 0.0;
            const double dz = dw * z * scale;
            sum_dz  += dz;
            sum_dz2 += dz * dz;
          }
          ame_out[m] += sum_dz;
          if (n_hon > 1) {
            const double v = (static_cast<double>(n_hon) * sum_dz2
                              - sum_dz * sum_dz)
                             / static_cast<double>(n_hon - 1);
            var_out[m] += std::max(v, 0.0);
          }
        }
      }
    }

#pragma omp critical
    {
      for (std::size_t idx = 0; idx < km; ++idx) {
        ame_acc[idx] += local_ame[idx];
        var_acc[idx] += local_var[idx];
      }
    }
  }
#else
  {
    std::vector<double> omega_plus(n_hon);
    std::vector<double> omega_minus(n_hon);
    std::vector<double> xi(k);

    for (int i = 0; i < n_eval; ++i) {
      const double* xi_src = x_rowmaj.data() + static_cast<std::ptrdiff_t>(i) * k;

      for (int jt = 0; jt < k_target; ++jt) {
        std::copy(xi_src, xi_src + k, xi.begin());
        const int col = t_col[jt];
        const double x_orig = xi[col];
        double scale;

        double x_plus_val, x_minus_val;
        if (t_disc[jt]) {
          x_minus_val = std::floor(x_orig);
          x_plus_val  = std::floor(x_orig) + 1.0;
          scale = 1.0;
        } else {
          x_minus_val = x_orig - t_h[jt];
          x_plus_val  = x_orig + t_h[jt];
          scale = 1.0 / (2.0 * t_h[jt]);
        }

        std::memset(omega_plus.data(), 0, n_hon * sizeof(double));
        std::memset(omega_minus.data(), 0, n_hon * sizeof(double));

        for (int b = 0; b < num_trees; ++b) {
          const HonestTree& ht = htrees[b];

          xi[col] = x_plus_val;
          const int leaf_plus = traverse_native(ht.td, xi);
          const int sp = ht.hon_offsets[leaf_plus];
          const int ep = ht.hon_offsets[leaf_plus + 1];
          const int lsz_p = ep - sp;
          if (lsz_p > 0) {
            const double w = inv_B / static_cast<double>(lsz_p);
            for (int pos = sp; pos < ep; ++pos)
              omega_plus[ht.hon_sorted[pos]] += w;
          }

          xi[col] = x_minus_val;
          const int leaf_minus = traverse_native(ht.td, xi);
          const int sm = ht.hon_offsets[leaf_minus];
          const int em = ht.hon_offsets[leaf_minus + 1];
          const int lsz_m = em - sm;
          if (lsz_m > 0) {
            const double w = inv_B / static_cast<double>(lsz_m);
            for (int pos = sm; pos < em; ++pos)
              omega_minus[ht.hon_sorted[pos]] += w;
          }
        }
        xi[col] = x_orig;

        double* ame_out = ame_acc.data() + static_cast<std::ptrdiff_t>(jt) * M;
        double* var_out = var_acc.data() + static_cast<std::ptrdiff_t>(jt) * M;
        for (int m = 0; m < M; ++m) {
          double sum_dz  = 0.0;
          double sum_dz2 = 0.0;
          for (int h = 0; h < n_hon; ++h) {
            const double dw = omega_plus[h] - omega_minus[h];
            if (dw == 0.0) continue;
            const double z = (y_hon_vec[h] == m) ? 1.0 : 0.0;
            const double dz = dw * z * scale;
            sum_dz  += dz;
            sum_dz2 += dz * dz;
          }
          ame_out[m] += sum_dz;
          if (n_hon > 1) {
            const double v = (static_cast<double>(n_hon) * sum_dz2
                              - sum_dz * sum_dz)
                             / static_cast<double>(n_hon - 1);
            var_out[m] += std::max(v, 0.0);
          }
        }
      }
    }
  }
#endif

  // Average over eval points
  const double n_eval_d = static_cast<double>(n_eval);
  const double n_eval_sq = n_eval_d * n_eval_d;

  Rcpp::NumericMatrix ame_r(k_target, M);
  Rcpp::NumericMatrix var_r(k_target, M);
  for (int jt = 0; jt < k_target; ++jt) {
    for (int m = 0; m < M; ++m) {
      const std::size_t idx = static_cast<std::size_t>(jt) * M + m;
      ame_r(jt, m) = ame_acc[idx] / n_eval_d;
      var_r(jt, m) = var_acc[idx] / n_eval_sq;
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("effects")  = ame_r,
    Rcpp::Named("variance") = var_r
  );
}
