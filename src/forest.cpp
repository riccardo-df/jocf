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

  // -------------------------------------------------------------------------
  // Phase 0 — Global pre-sort (sequential, done once for entire forest)
  // Compute rank array index_data[j * n + i] and unique_values[j].
  // Cost: O(k * n * log n), amortized over all B trees.
  // -------------------------------------------------------------------------
  SortData sort_data;
  sort_data.n    = n;
  sort_data.k    = k;
  sort_data.x_cm = x.memptr();
  sort_data.index_data.resize(static_cast<std::size_t>(k) * n);
  sort_data.unique_values.resize(k);
  sort_data.n_unique.resize(k);

  {
    std::vector<std::pair<double, int>> val_idx(n);
    for (int j = 0; j < k; ++j) {
      const double* col = sort_data.x_cm + static_cast<std::ptrdiff_t>(j) * n;
      for (int i = 0; i < n; ++i) val_idx[i] = {col[i], i};
      std::sort(val_idx.begin(), val_idx.end());

      sort_data.unique_values[j].clear();
      sort_data.unique_values[j].push_back(val_idx[0].first);
      int rank = 0;
      sort_data.index_data[static_cast<std::ptrdiff_t>(j) * n
                           + val_idx[0].second] = 0;
      for (int i = 1; i < n; ++i) {
        if (val_idx[i].first > val_idx[i - 1].first) {
          sort_data.unique_values[j].push_back(val_idx[i].first);
          ++rank;
        }
        sort_data.index_data[static_cast<std::ptrdiff_t>(j) * n
                             + val_idx[i].second] = rank;
      }
      sort_data.n_unique[j] = rank + 1;
    }
  }

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
