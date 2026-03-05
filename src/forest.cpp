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
// Template accepts arma::rowvec, std::vector<double>, or anything
// supporting operator[].  Read-only; thread-safe.
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
//' @param mtry Number of candidate features at each split.
//' @param M Number of outcome classes.
//' @param lambda Numeric weight vector of length M.
//' @param num_threads Number of OpenMP threads (0 = all available).
//'
//' @return Named list: `predictions` (n x M) and `forest` (list of B trees).
//' @export
// [[Rcpp::export]]
Rcpp::List grow_forest_cpp(
  Rcpp::IntegerVector Y,
  Rcpp::NumericMatrix X,
  int                 num_trees,
  int                 min_node_size,
  int                 mtry,
  int                 M,
  Rcpp::NumericVector lambda,
  int                 num_threads = 0
) {
  const int n = Y.size();

  // Convert Y to 0-indexed arma::ivec
  arma::ivec y(n);
  for (int i = 0; i < n; ++i) y[i] = Y[i] - 1;

  const arma::mat x   = Rcpp::as<arma::mat>(X);
  const arma::vec lam = Rcpp::as<arma::vec>(lambda);

  // -------------------------------------------------------------------------
  // Phase 1 — sequential, R's RNG
  // Generate per-tree std::mt19937 seeds and bootstrap index arrays.
  // All R API calls are confined to this phase so the parallel phase is safe.
  // -------------------------------------------------------------------------
  Rcpp::NumericVector seed_doubles = Rcpp::runif(num_trees, 0.0, 4294967295.0);
  std::vector<uint32_t> tree_seeds(num_trees);
  for (int b = 0; b < num_trees; ++b)
    tree_seeds[b] = static_cast<uint32_t>(seed_doubles[b]);

  std::vector<arma::uvec> boot_indices(num_trees);
  for (int b = 0; b < num_trees; ++b) {
    Rcpp::IntegerVector boot_r = Rcpp::sample(n, n, /*replace=*/true) - 1;
    boot_indices[b] = Rcpp::as<arma::uvec>(boot_r);
  }

  // -------------------------------------------------------------------------
  // Phase 2 — parallel over trees
  // -------------------------------------------------------------------------
  std::vector<TreeData> native_forest(num_trees);
  arma::mat pred_acc(n, M, arma::fill::zeros);

#ifdef _OPENMP
  const int nt_grow = (num_threads > 0) ? num_threads : omp_get_max_threads();
#pragma omp parallel num_threads(nt_grow)
  {
    arma::mat local_pred(n, M, arma::fill::zeros);

#pragma omp for schedule(dynamic)
    for (int b = 0; b < num_trees; ++b) {
      std::mt19937 rng(tree_seeds[b]);
      native_forest[b] = grow_single_tree(y, x, boot_indices[b], M, lam,
                                           min_node_size, mtry, rng);
      const TreeData& td = native_forest[b];
      for (int i = 0; i < n; ++i) {
        const arma::rowvec xi = x.row(i);
        const int leaf = traverse_native(td, xi);
        for (int m = 0; m < M; ++m)
          local_pred(i, m) += td.leaf_probs[leaf * M + m];
      }
    }

#pragma omp critical
    pred_acc += local_pred;
  }
#else
  for (int b = 0; b < num_trees; ++b) {
    std::mt19937 rng(tree_seeds[b]);
    native_forest[b] = grow_single_tree(y, x, boot_indices[b], M, lam,
                                         min_node_size, mtry, rng);
    const TreeData& td = native_forest[b];
    for (int i = 0; i < n; ++i) {
      const arma::rowvec xi = x.row(i);
      const int leaf = traverse_native(td, xi);
      for (int m = 0; m < M; ++m)
        pred_acc(i, m) += td.leaf_probs[leaf * M + m];
    }
  }
#endif

  // -------------------------------------------------------------------------
  // Phase 3 — sequential: serialise native trees to R lists
  // -------------------------------------------------------------------------
  Rcpp::List forest_r(num_trees);
  for (int b = 0; b < num_trees; ++b)
    forest_r[b] = tree_to_list(native_forest[b], M);

  const arma::mat predictions = pred_acc / static_cast<double>(num_trees);
  return Rcpp::List::create(
    Rcpp::Named("predictions") = Rcpp::wrap(predictions),
    Rcpp::Named("forest")      = forest_r
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
//' @return Numeric matrix (n_test x M) of predicted class probabilities.
//' @export
// [[Rcpp::export]]
Rcpp::NumericMatrix predict_forest_cpp(
  Rcpp::List          forest,
  Rcpp::NumericMatrix X_new,
  int                 M,
  int                 num_threads = 0
) {
  const int n_test    = X_new.nrow();
  const int num_trees = static_cast<int>(forest.size());

  const arma::mat x_new = Rcpp::as<arma::mat>(X_new);

  // Deserialise all trees before entering the parallel region
  std::vector<TreeData> native_forest(num_trees);
  for (int b = 0; b < num_trees; ++b)
    native_forest[b] = list_to_tree(Rcpp::as<Rcpp::List>(forest[b]), M);

  arma::mat pred_acc(n_test, M, arma::fill::zeros);

#ifdef _OPENMP
  const int nt_pred = (num_threads > 0) ? num_threads : omp_get_max_threads();
#pragma omp parallel num_threads(nt_pred)
  {
    arma::mat local_pred(n_test, M, arma::fill::zeros);

#pragma omp for schedule(dynamic)
    for (int b = 0; b < num_trees; ++b) {
      const TreeData& td = native_forest[b];
      for (int i = 0; i < n_test; ++i) {
        const arma::rowvec xi = x_new.row(i);
        const int leaf = traverse_native(td, xi);
        for (int m = 0; m < M; ++m)
          local_pred(i, m) += td.leaf_probs[leaf * M + m];
      }
    }

#pragma omp critical
    pred_acc += local_pred;
  }
#else
  for (int b = 0; b < num_trees; ++b) {
    const TreeData& td = native_forest[b];
    for (int i = 0; i < n_test; ++i) {
      const arma::rowvec xi = x_new.row(i);
      const int leaf = traverse_native(td, xi);
      for (int m = 0; m < M; ++m)
        pred_acc(i, m) += td.leaf_probs[leaf * M + m];
    }
  }
#endif

  const arma::mat predictions = pred_acc / static_cast<double>(num_trees);
  return Rcpp::wrap(predictions);
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
      const TreeData& td = native_forest[b];

      for (int i = 0; i < n_eval; ++i) {
        // Copy row i of X_eval into xi (modifiable scratch vector)
        for (int j = 0; j < k; ++j) xi[j] = x_eval_arma(i, j);

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
          for (int m = 0; m < M; ++m)
            local_eff[jt * M + m] +=
              (td.leaf_probs[base_hi + m] - td.leaf_probs[base_lo + m]) * scale;
        }
      }
    }

#pragma omp critical
    for (int i = 0; i < k_target * M; ++i)
      effects_acc[i] += local_eff[i];
  }
#else
  {
    std::vector<double> xi(k);
    for (int b = 0; b < num_trees; ++b) {
      const TreeData& td = native_forest[b];
      for (int i = 0; i < n_eval; ++i) {
        for (int j = 0; j < k; ++j) xi[j] = x_eval_arma(i, j);

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
          for (int m = 0; m < M; ++m)
            effects_acc[jt * M + m] +=
              (td.leaf_probs[base_hi + m] - td.leaf_probs[base_lo + m]) * scale;
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
      result(jt, m) = effects_acc[jt * M + m];

  return result;
}
