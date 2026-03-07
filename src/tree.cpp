// tree.cpp
// Tier 4: grow_single_tree() — BFS tree growing with in-place partition.
//
// Returns a native TreeData struct (thread-safe, no R API).
// The caller (grow_forest_cpp) supplies a seeded std::mt19937 for the
// partial Fisher-Yates feature shuffle, allowing fully deterministic
// parallel tree growing when seeds are pre-generated in R's RNG stream.
//
// Architecture:
//   - No per-tree data copy: trees work on original X via SortData indirection.
//   - In-place partition: each tree has a single sample_ids vector (bootstrap
//     indices). Nodes tracked by start_pos/end_pos. Splits partition via swap.
//   - Ranked split search: bucket observations by pre-computed rank, sweep
//     unique values.  No per-node sorting.
//   - BFS growing: nodes processed 0, 1, 2, ... sequentially. Children
//     appended to end of arrays.
//   - Zero per-node heap allocation.

#include "jocf_internal.h"
#include <vector>
#include <algorithm>
#include <numeric>
// [[Rcpp::depends(RcppArmadillo)]]

// ---------------------------------------------------------------------------
// grow_single_tree  (declared in jocf_internal.h)
// ---------------------------------------------------------------------------
TreeData grow_single_tree(
  const int*        y,           // 0-indexed class labels, full dataset
  const SortData&   sort_data,   // global pre-sort (read-only)
  const int*        boot_idx,    // bootstrap indices (0-based, length n_boot)
  int               n_boot,
  int               M,
  const double*     lambda,
  int               min_node_size,
  int               mtry,
  std::mt19937&     rng,
  int               max_depth    // -1 = unlimited
) {
  const int n = sort_data.n;
  const int k = sort_data.k;
  const int mtry_k = std::min(mtry, k);

  // Compute max_n_unique across all features for scratch buffer sizing
  int max_n_unique = 0;
  for (int j = 0; j < k; ++j)
    if (sort_data.n_unique[j] > max_n_unique)
      max_n_unique = sort_data.n_unique[j];

  // ----- Per-tree buffers (zero per-node allocation) -----------------------
  // sample_ids: mutable copy of bootstrap indices, partitioned in-place
  std::vector<int> sample_ids(boot_idx, boot_idx + n_boot);

  // Scratch for ranked split search (reused across all nodes)
  std::vector<int> counter(max_n_unique);
  std::vector<int> counter_pc(static_cast<std::size_t>(max_n_unique) * M);

  // Feature permutation for Fisher-Yates mtry selection
  std::vector<int> perm(k);
  std::vector<int> feat_sub(mtry_k);

  // Per-node class counts (reused)
  std::vector<int> node_counts(M);

  // ----- Tree storage (dynamic, grown node by node) -----------------------
  // Reserve based on expected tree size
  const int reserve_sz = std::max(2 * n_boot / std::max(min_node_size, 1) + 1, 64);

  std::vector<int>    sf;   // split_feature  (1-based, 0 = leaf)
  std::vector<double> st;   // split_threshold
  std::vector<int>    lc;   // left_child  (0-based, -1 = leaf)
  std::vector<int>    rc;   // right_child (0-based, -1 = leaf)
  std::vector<double> lp;   // leaf_probs row-major: [node_id * M + m]
  std::vector<int>    start_pos;  // node start in sample_ids (inclusive)
  std::vector<int>    end_pos;    // node end in sample_ids (exclusive)
  std::vector<int>    node_depth; // depth of each node (root = 0)

  sf.reserve(reserve_sz);
  st.reserve(reserve_sz);
  lc.reserve(reserve_sz);
  rc.reserve(reserve_sz);
  lp.reserve(static_cast<std::size_t>(reserve_sz) * M);
  start_pos.reserve(reserve_sz);
  end_pos.reserve(reserve_sz);
  node_depth.reserve(reserve_sz);

  // Allocate one new node; returns its 0-based index.
  auto alloc_node = [&](int s, int e, int depth) -> int {
    int id = static_cast<int>(sf.size());
    sf.push_back(0);
    st.push_back(0.0);
    lc.push_back(-1);
    rc.push_back(-1);
    for (int m = 0; m < M; ++m) lp.push_back(0.0);
    start_pos.push_back(s);
    end_pos.push_back(e);
    node_depth.push_back(depth);
    return id;
  };

  // ----- BFS growing -------------------------------------------------------
  alloc_node(0, n_boot, 0);  // root node (depth 0)
  int n_nodes = 1;
  int cursor  = 0;

  while (cursor < n_nodes) {
    const int s     = start_pos[cursor];
    const int e     = end_pos[cursor];
    const int n_obs = e - s;

    // Class counts for this node
    std::fill(node_counts.begin(), node_counts.end(), 0);
    for (int i = s; i < e; ++i) node_counts[y[sample_ids[i]]]++;

    // Check depth limit
    if (max_depth > 0 && node_depth[cursor] >= max_depth) {
      for (int m = 0; m < M; ++m)
        lp[cursor * M + m] = static_cast<double>(node_counts[m]) / n_obs;
      ++cursor;
      continue;
    }

    // Check if too small to split
    if (n_obs < 2 * min_node_size) {
      // Set leaf probabilities
      for (int m = 0; m < M; ++m)
        lp[cursor * M + m] = static_cast<double>(node_counts[m]) / n_obs;
      ++cursor;
      continue;
    }

    // Partial Fisher-Yates to select mtry features without replacement
    std::iota(perm.begin(), perm.end(), 0);
    for (int i = 0; i < mtry_k; ++i) {
      std::uniform_int_distribution<int> d(i, k - 1);
      std::swap(perm[i], perm[d(rng)]);
    }
    for (int i = 0; i < mtry_k; ++i) feat_sub[i] = perm[i];

    // Ranked split search
    SplitResult sr = find_best_split_ranked(
      y, sort_data,
      sample_ids.data(), s, e,
      M, lambda, min_node_size,
      feat_sub.data(), mtry_k,
      counter, counter_pc
    );

    if (!sr.found) {
      // Set leaf probabilities
      for (int m = 0; m < M; ++m)
        lp[cursor * M + m] = static_cast<double>(node_counts[m]) / n_obs;
      ++cursor;
      continue;
    }

    // Record split
    sf[cursor] = sr.feature;
    st[cursor] = sr.threshold;

    // In-place partition of sample_ids[s..e-1] by RANK
    // Observations with rank <= sr.rank go left.
    const int split_col = sr.feature - 1;  // 0-based
    const int* rank_col = sort_data.index_data.data()
                          + static_cast<std::ptrdiff_t>(split_col) * n;

    int mid = s;
    for (int i = s; i < e; ++i) {
      if (rank_col[sample_ids[i]] <= sr.rank) {
        std::swap(sample_ids[i], sample_ids[mid]);
        ++mid;
      }
    }

    // Allocate children
    int child_depth = node_depth[cursor] + 1;
    int left_id  = alloc_node(s, mid, child_depth);
    int right_id = alloc_node(mid, e, child_depth);
    lc[cursor] = left_id;
    rc[cursor] = right_id;
    n_nodes += 2;

    ++cursor;
  }

  // ----- Fill and return TreeData -----------------------------------------
  TreeData td;
  td.n_nodes         = n_nodes;
  td.split_feature   = std::move(sf);
  td.split_threshold = std::move(st);
  td.left_child      = std::move(lc);
  td.right_child     = std::move(rc);
  td.leaf_probs      = std::move(lp);
  return td;
}
