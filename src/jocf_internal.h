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
// compute_impurity
// Weighted Gini impurity for a node with given class counts.
//   Q(C) = (1/M) * sum_m  lambda[m] * p_m * (1 - p_m)
// Defined inline so each including TU gets its own copy (ODR-safe).
// ---------------------------------------------------------------------------
inline double compute_impurity(const arma::ivec& counts,
                                int              n,
                                int              M,
                                const arma::vec& lambda) {
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
// SplitResult: output of find_best_split_internal
// ---------------------------------------------------------------------------
struct SplitResult {
  int    feature;    // 1-based column index; 0 if no valid split found
  double threshold;  // split value; obs goes left iff x[feat] <= threshold
  double impurity;   // Q(left) + Q(right) at best split
  bool   found;      // true iff feature > 0
};

// ---------------------------------------------------------------------------
// find_best_split_internal  (implemented in splitting.cpp)
//
// Exhaustive split search over feature_subset for a single node.
// ---------------------------------------------------------------------------
SplitResult find_best_split_internal(
  const arma::ivec& y,
  const arma::mat&  x,
  int               M,
  const arma::vec&  lambda,
  int               min_node_size,
  const arma::uvec& feature_subset
);

// ---------------------------------------------------------------------------
// grow_single_tree  (implemented in tree.cpp)
//
// Returns a native TreeData object (thread-safe, no R API calls).
// rng is the per-tree Mersenne Twister, seeded by grow_forest_cpp before
// the parallel region.
// ---------------------------------------------------------------------------
TreeData grow_single_tree(
  const arma::ivec& y,
  const arma::mat&  x,
  const arma::uvec& boot_idx,
  int               M,
  const arma::vec&  lambda,
  int               min_node_size,
  int               mtry,
  std::mt19937&     rng
);
