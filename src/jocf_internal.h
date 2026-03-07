// jocf_internal.h
// Internal declarations and inline helpers shared across compilation units.
// Include only in .cpp files (never in user-facing headers).

#pragma once
#include <RcppArmadillo.h>
#include <random>
#include <vector>
#include <numeric>
#include <limits>

// ---------------------------------------------------------------------------
// TreeData: C++ native tree representation (no R API, thread-safe).
// leaf_probs is stored row-major: leaf_probs[node * M + m].
// ---------------------------------------------------------------------------
struct TreeData {
  std::vector<int>    split_feature;   // 1-based, 0 = leaf
  std::vector<double> split_threshold;
  std::vector<int>    left_child;      // 0-based, -1 = leaf
  std::vector<int>    right_child;
  std::vector<double> leaf_probs;      // row-major: [node * M + m]
  int n_nodes = 0;
};

// ---------------------------------------------------------------------------
// SortData: global pre-sort structure, computed once per forest.
// index_data[j * n + i] = rank of observation i for feature j (0-based).
// unique_values[j] = sorted unique values for feature j.
// n_unique[j] = number of unique values for feature j.
// All trees share a single read-only SortData.
// ---------------------------------------------------------------------------
struct SortData {
  int                n;                // total observations in dataset
  int                k;                // number of features
  const double*      x_cm;            // original X, col-major [j*n + i]
  std::vector<int>   index_data;      // ranks: [j * n + i]
  std::vector<std::vector<double>> unique_values;  // unique_values[j] = sorted
  std::vector<int>   n_unique;        // n_unique[j] = #unique for col j
};

// ---------------------------------------------------------------------------
// compute_impurity
// Weighted Gini impurity for a node with given class counts.
//   Q(C) = (1/M) * sum_m  lambda[m] * p_m * (1 - p_m)
// Raw pointer interface — no Armadillo allocation in the hot path.
// Defined inline so each including TU gets its own copy (ODR-safe).
// ---------------------------------------------------------------------------
inline double compute_impurity(const int*    counts,
                                int           n,
                                int           M,
                                const double* lambda) {
  if (n <= 0) return 0.0;
  const double nd = static_cast<double>(n);
  double Q = 0.0;
  for (int m = 0; m < M; ++m) {
    const double p = counts[m] / nd;
    Q += lambda[m] * p * (1.0 - p);
  }
  return Q / static_cast<double>(M);
}

// ---------------------------------------------------------------------------
// SplitResult: output of find_best_split_internal / find_best_split_ranked
// ---------------------------------------------------------------------------
struct SplitResult {
  int    feature;    // 1-based column index; 0 if no valid split found
  double threshold;  // split value; obs goes left iff x[feat] <= threshold
  double impurity;   // Q(left) + Q(right) at best split
  bool   found;      // true iff feature > 0
  int    rank;       // 0-based rank of split point; obs left iff rank <= this
                     // -1 if unused (find_best_split_internal)
};

// ---------------------------------------------------------------------------
// find_best_split_internal  (implemented in splitting.cpp)
//
// Exhaustive split search over feat_sub for a single node.
//
// Raw pointer interface — avoids Armadillo submatrix allocations.
// Only the mtry candidate columns are copied (not all k), saving work when
// mtry << k.  val_buf holds a contiguous copy of one column's values for the
// current node, enabling a cache-friendly sort (single dereference comparator).
//
// Parameters:
//   y_boot     — 0-indexed class labels for the full bootstrap sample (length n_boot)
//   x_boot_cm  — covariate matrix for the full bootstrap sample, column-major
//                layout: element (row i, col j) is at x_boot_cm[j * n_boot + i]
//   n_boot     — number of rows in x_boot_cm (used as column stride)
//   obs        — indices (into y_boot / x_boot_cm rows) for the current node
//   n_obs      — number of observations in this node
//   M          — number of outcome classes
//   lambda     — weight vector of length M (1.0 for unweighted)
//   min_node_size — minimum observations per child after split
//   feat_sub   — 0-based column indices to consider (length n_feat_sub)
//   n_feat_sub — number of candidate features (mtry)
//   sort_buf   — integer scratch buffer; reused across calls
//   val_buf    — double scratch buffer; holds one column's node values for sort
// ---------------------------------------------------------------------------
SplitResult find_best_split_internal(
  const int*         y_boot,
  const double*      x_boot_cm,
  int                n_boot,
  const int*         obs,
  int                n_obs,
  int                M,
  const double*      lambda,
  int                min_node_size,
  const int*         feat_sub,
  int                n_feat_sub,
  std::vector<int>&    sort_buf,
  std::vector<double>& val_buf
);

// ---------------------------------------------------------------------------
// find_best_split_ranked  (implemented in splitting.cpp)
//
// Rank-based split search: buckets observations by pre-computed rank, then
// sweeps unique values.  No per-node sorting required.
//
// Parameters:
//   y          — 0-indexed class labels for the full dataset
//   sort_data  — global pre-sort structure (read-only)
//   sample_ids — bootstrap indices (0-based) for the current tree
//   start      — start index into sample_ids for this node (inclusive)
//   end        — end index into sample_ids for this node (exclusive)
//   M          — number of outcome classes
//   lambda     — weight vector of length M
//   min_node_size — minimum observations per child after split
//   feat_sub   — 0-based column indices to consider (length n_feat_sub)
//   n_feat_sub — number of candidate features (mtry)
//   counter    — scratch buffer [max_n_unique]; reused across calls
//   counter_pc — scratch buffer [max_n_unique * M]; reused across calls
// ---------------------------------------------------------------------------
SplitResult find_best_split_ranked(
  const int*        y,
  const SortData&   sort_data,
  const int*        sample_ids,
  int               start,
  int               end,
  int               M,
  const double*     lambda,
  int               min_node_size,
  const int*        feat_sub,
  int               n_feat_sub,
  std::vector<int>& counter,
  std::vector<int>& counter_pc
);

// ---------------------------------------------------------------------------
// grow_single_tree  (implemented in tree.cpp)
//
// BFS tree growing with in-place partition and ranked split search.
// Returns a native TreeData object (thread-safe, no R API calls).
// rng is the per-tree Mersenne Twister, seeded by grow_forest_cpp before
// the parallel region.
//
// Parameters:
//   y          — 0-indexed class labels, full dataset (length sort_data.n)
//   sort_data  — global pre-sort structure (read-only, shared across trees)
//   boot_idx   — bootstrap indices (0-based, length n_boot)
//   n_boot     — bootstrap sample size
//   M          — number of outcome classes
//   lambda     — weight vector of length M
//   min_node_size — minimum observations per terminal node
//   mtry       — number of candidate features per split
//   rng        — per-tree Mersenne Twister
//   max_depth  — maximum tree depth (-1 = unlimited)
// ---------------------------------------------------------------------------
TreeData grow_single_tree(
  const int*        y,
  const SortData&   sort_data,
  const int*        boot_idx,
  int               n_boot,
  int               M,
  const double*     lambda,
  int               min_node_size,
  int               mtry,
  std::mt19937&     rng,
  int               max_depth = -1
);
