// tree.cpp
// Phase 2: grow_single_tree() — iterative binary partitioner for jocf.
//
// Returns a native TreeData struct (thread-safe, no R API).
// The caller (grow_forest_cpp) supplies a seeded std::mt19937 for the
// partial Fisher-Yates feature shuffle, allowing fully deterministic
// parallel tree growing when seeds are pre-generated in R's RNG stream.

#include "jocf_internal.h"
#include <vector>
#include <stack>
#include <numeric>
#include <algorithm>
// [[Rcpp::depends(RcppArmadillo)]]

// ---------------------------------------------------------------------------
// Internal stack entry for iterative tree growing.
// ---------------------------------------------------------------------------
struct StackEntry {
  int        node_id;
  arma::uvec obs_local_idx;
};

// ---------------------------------------------------------------------------
// grow_single_tree  (declared in jocf_internal.h)
// ---------------------------------------------------------------------------
TreeData grow_single_tree(
  const arma::ivec& y,           // 0-indexed class labels, full dataset
  const arma::mat&  x,           // full covariate matrix (n x k)
  const arma::uvec& boot_idx,    // bootstrap sample (0-based row indices)
  int               M,
  const arma::vec&  lambda,
  int               min_node_size,
  int               mtry,
  std::mt19937&     rng
) {
  const int n_boot = static_cast<int>(boot_idx.n_elem);
  const int k      = static_cast<int>(x.n_cols);
  const int mtry_k = std::min(mtry, k);

  // Build the bootstrap subsample once (avoids repeated indexing later)
  arma::ivec y_boot(n_boot);
  arma::mat  x_boot(n_boot, k);
  for (int i = 0; i < n_boot; ++i) {
    y_boot[i]     = y[boot_idx[i]];
    x_boot.row(i) = x.row(boot_idx[i]);
  }

  // ----- Tree storage (dynamic, grown node by node) -----------------------
  std::vector<int>    sf;   // split_feature  (1-based, 0 = leaf)
  std::vector<double> st;   // split_threshold
  std::vector<int>    lc;   // left_child  (0-based, -1 = leaf)
  std::vector<int>    rc;   // right_child (0-based, -1 = leaf)
  // leaf_probs stored row-major: lp[node_id * M + m]
  std::vector<double> lp;

  // Allocate one new node; returns its 0-based index.
  auto alloc_node = [&]() -> int {
    int id = static_cast<int>(sf.size());
    sf.push_back(0);
    st.push_back(0.0);
    lc.push_back(-1);
    rc.push_back(-1);
    for (int m = 0; m < M; ++m) lp.push_back(0.0);
    return id;
  };

  // Write class proportions for a leaf node.
  auto set_leaf = [&](int node_id, const arma::ivec& counts, int n_node) {
    for (int m = 0; m < M; ++m)
      lp[node_id * M + m] = static_cast<double>(counts[m]) / n_node;
  };

  // ----- Iterative growing ------------------------------------------------
  std::stack<StackEntry> stk;
  int root_id = alloc_node();
  stk.push({root_id, arma::regspace<arma::uvec>(0, n_boot - 1)});

  // Permutation vector reused across nodes for partial Fisher-Yates
  std::vector<int> perm(k);

  while (!stk.empty()) {
    StackEntry entry = std::move(stk.top());
    stk.pop();

    const int node_id = entry.node_id;
    arma::uvec& obs   = entry.obs_local_idx;
    const int n_node  = static_cast<int>(obs.n_elem);

    // Compute class counts for this node
    arma::ivec counts(M, arma::fill::zeros);
    for (int i = 0; i < n_node; ++i) counts[y_boot[obs[i]]]++;

    // Stopping condition: too few observations to split
    if (n_node < 2 * min_node_size) {
      set_leaf(node_id, counts, n_node);
      continue;
    }

    // Partial Fisher-Yates to select mtry features without replacement.
    // Uses the per-tree rng — no R API call, safe inside OpenMP parallel region.
    std::iota(perm.begin(), perm.end(), 0);
    for (int i = 0; i < mtry_k; ++i) {
      std::uniform_int_distribution<int> d(i, k - 1);
      std::swap(perm[i], perm[d(rng)]);
    }
    arma::uvec feat_sub(mtry_k);
    for (int i = 0; i < mtry_k; ++i) feat_sub[i] = static_cast<arma::uword>(perm[i]);

    // Extract node sub-data
    arma::mat  x_node = x_boot.rows(obs);   // (n_node x k)
    arma::ivec y_node = y_boot.elem(obs);   // (n_node)

    SplitResult sr = find_best_split_internal(y_node, x_node, M, lambda,
                                               min_node_size, feat_sub);

    if (!sr.found) {
      set_leaf(node_id, counts, n_node);
      continue;
    }

    // Record split in current node
    sf[node_id] = sr.feature;
    st[node_id] = sr.threshold;

    // Partition local observations into left and right children
    const int    feat_0  = sr.feature - 1;    // 0-based column
    arma::vec    xj_node = x_node.col(feat_0);
    arma::uvec   left_mask  = arma::find(xj_node <= sr.threshold);
    arma::uvec   right_mask = arma::find(xj_node >  sr.threshold);

    arma::uvec left_obs  = obs.elem(left_mask);
    arma::uvec right_obs = obs.elem(right_mask);

    // Allocate child nodes and wire parent -> children
    int left_id  = alloc_node();
    int right_id = alloc_node();
    lc[node_id] = left_id;
    rc[node_id] = right_id;

    stk.push({left_id,  std::move(left_obs)});
    stk.push({right_id, std::move(right_obs)});
  }

  // ----- Fill and return TreeData -----------------------------------------
  TreeData td;
  td.n_nodes         = static_cast<int>(sf.size());
  td.split_feature   = std::move(sf);
  td.split_threshold = std::move(st);
  td.left_child      = std::move(lc);
  td.right_child     = std::move(rc);
  td.leaf_probs      = std::move(lp);
  return td;
}
