// splitting.cpp
// Phase 1: node impurity and best-split search for the joint OCF.
//
// Splitting criterion (unweighted):
//   Q(C) = (1/M) * sum_{m=1}^{M}  p_m * (1 - p_m)
//   p_m  = count(Y_i = m in C) / |C|
//
// Splitting criterion (variance-weighted):
//   Q_w(C) = (1/M) * sum_{m=1}^{M}  lambda_m * p_m * (1 - p_m)
//   lambda_m = 1 / (p_hat_m * (1 - p_hat_m)),  p_hat_m = global proportion
//
// Pass lambda = rep(1, M) to recover the unweighted criterion.
// Exported functions (node_impurity_cpp, find_best_split_cpp) are used in
// unit tests.  Internal function find_best_split_internal is called by tree.cpp.
//
// Tier 1 optimisation: all inner-loop Armadillo objects replaced with raw
// C++ pointers and std::vector.  No submatrix extraction; obs indices are
// passed directly and x_boot_cm is accessed via column-major pointer arithmetic.

#include "jocf_internal.h"
#include <algorithm>
#include <numeric>
// [[Rcpp::depends(RcppArmadillo)]]

// ---------------------------------------------------------------------------
// find_best_split_internal  (declared in jocf_internal.h)
//
// Searches feat_sub for the split minimising Q(left) + Q(right).
// y_boot is 0-indexed (values 0..M-1).
//
// x_boot_cm is column-major with n_boot rows:
//   element at (row, col) = x_boot_cm[col * n_boot + row]
//
// obs[0..n_obs-1] are row indices into y_boot / x_boot_cm for this node.
// sort_buf is resized as needed and reused across calls to avoid allocation.
// ---------------------------------------------------------------------------
SplitResult find_best_split_internal(
  const int*           y_boot,
  const double*        x_boot_cm,
  int                  n_boot,
  const int*           obs,
  int                  n_obs,
  int                  M,
  const double*        lambda,
  int                  min_node_size,
  const int*           feat_sub,
  int                  n_feat_sub,
  std::vector<int>&    sort_buf,
  std::vector<double>& val_buf
) {
  // Total class counts for this node
  std::vector<int> total_counts(M, 0);
  for (int i = 0; i < n_obs; ++i) total_counts[y_boot[obs[i]]]++;

  double best_impurity  = std::numeric_limits<double>::infinity();
  int    best_feature   = 0;        // 1-based; 0 = not found
  double best_threshold = 0.0;

  std::vector<int> left_counts(M), right_counts(M);

  // Scratch buffers resized once per node (reused across features)
  sort_buf.resize(n_obs);
  val_buf.resize(n_obs);

  for (int fi = 0; fi < n_feat_sub; ++fi) {
    const int     j      = feat_sub[fi];          // 0-based column
    const double* xj_col = x_boot_cm + static_cast<std::ptrdiff_t>(j) * n_boot;

    // Copy node's feature values to a contiguous buffer.
    // This turns the subsequent sort comparator into a single dereference
    // (cache-friendly) rather than a double-indirection random access.
    for (int i = 0; i < n_obs; ++i) val_buf[i] = xj_col[obs[i]];

    // Sort local indices 0..n_obs-1 by contiguous val_buf values
    std::iota(sort_buf.begin(), sort_buf.end(), 0);
    std::sort(sort_buf.begin(), sort_buf.end(), [&](int a, int b) {
      return val_buf[a] < val_buf[b];
    });

    std::fill(left_counts.begin(), left_counts.end(), 0);
    right_counts = total_counts;
    int n_left = 0, n_right = n_obs;

    for (int ii = 0; ii < n_obs - 1; ++ii) {
      const int local_i  = sort_buf[ii];
      const int m        = y_boot[obs[local_i]];

      left_counts[m]++;
      right_counts[m]--;
      ++n_left;
      --n_right;

      // No split between identical feature values
      const double xval_cur  = val_buf[sort_buf[ii]];
      const double xval_next = val_buf[sort_buf[ii + 1]];
      if (xval_cur >= xval_next) continue;

      // Enforce minimum node size on both children
      if (n_left  < min_node_size) continue;
      if (n_right < min_node_size) continue;

      const double imp =
          compute_impurity(left_counts.data(),  n_left,  M, lambda) +
          compute_impurity(right_counts.data(), n_right, M, lambda);

      if (imp < best_impurity) {
        best_impurity  = imp;
        best_feature   = j + 1;    // convert to 1-based
        best_threshold = 0.5 * (xval_cur + xval_next);
      }
    }
  }

  return {best_feature, best_threshold, best_impurity, (best_feature > 0), -1};
}

// ---------------------------------------------------------------------------
// find_best_split_ranked  (declared in jocf_internal.h)
//
// Rank-based split search: buckets observations by pre-computed rank,
// then sweeps unique values.  No per-node sorting required.
//
// Algorithm per candidate feature j:
//   1. Clear counter[0..n_unique_j-1] and counter_pc[0..n_unique_j*M-1]
//   2. Bucket: for each obs in sample_ids[start..end-1]:
//        r = index_data[j*n + obs];  counter[r]++;  counter_pc[r*M + y[obs]]++
//   3. Sweep unique values r = 0..n_unique_j-2:
//        Move counter[r] obs from right to left; compute impurity; track best.
//
// Complexity per feature: O(n_node + n_unique_j * M).
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
) {
  const int n_obs = end - start;
  const int n     = sort_data.n;

  // Total class counts for this node
  std::vector<int> total_counts(M, 0);
  for (int i = start; i < end; ++i) total_counts[y[sample_ids[i]]]++;

  double best_impurity  = std::numeric_limits<double>::infinity();
  int    best_feature   = 0;        // 1-based; 0 = not found
  double best_threshold = 0.0;
  int    best_rank      = -1;

  std::vector<int> left_counts(M), right_counts(M);

  for (int fi = 0; fi < n_feat_sub; ++fi) {
    const int j        = feat_sub[fi];             // 0-based column
    const int n_uniq_j = sort_data.n_unique[j];
    const int* rank_col = sort_data.index_data.data()
                          + static_cast<std::ptrdiff_t>(j) * n;

    // Clear counters for this feature
    std::fill(counter.begin(), counter.begin() + n_uniq_j, 0);
    std::fill(counter_pc.begin(), counter_pc.begin() + n_uniq_j * M, 0);

    // Bucket observations by rank
    for (int i = start; i < end; ++i) {
      const int obs = sample_ids[i];
      const int r   = rank_col[obs];
      counter[r]++;
      counter_pc[r * M + y[obs]]++;
    }

    // Sweep unique values left-to-right
    std::fill(left_counts.begin(), left_counts.end(), 0);
    right_counts = total_counts;
    int n_left = 0, n_right = n_obs;

    for (int r = 0; r < n_uniq_j - 1; ++r) {
      // Move counter[r] obs from right to left
      n_left  += counter[r];
      n_right -= counter[r];
      for (int m = 0; m < M; ++m) {
        left_counts[m]  += counter_pc[r * M + m];
        right_counts[m] -= counter_pc[r * M + m];
      }

      // Skip if no observations at this rank
      if (counter[r] == 0) continue;

      // Enforce minimum node size
      if (n_left < min_node_size) continue;
      if (n_right < min_node_size) break;   // early termination

      const double imp =
          compute_impurity(left_counts.data(),  n_left,  M, lambda) +
          compute_impurity(right_counts.data(), n_right, M, lambda);

      if (imp < best_impurity) {
        best_impurity  = imp;
        best_feature   = j + 1;    // convert to 1-based
        best_threshold = 0.5 * (sort_data.unique_values[j][r]
                              + sort_data.unique_values[j][r + 1]);
        best_rank      = r;
      }
    }
  }

  return {best_feature, best_threshold, best_impurity, (best_feature > 0),
          best_rank};
}

// ---------------------------------------------------------------------------
// node_impurity_cpp
// Exported for unit testing; thin wrapper around compute_impurity().
//
// Parameters
//   counts  : integer vector of length M; counts[m] = #obs with Y == m+1
//   n       : total observations in the node  (= sum(counts))
//   M       : number of outcome classes
//   lambda  : numeric vector of length M; use rep(1, M) for unweighted
//
// Returns the scalar impurity Q(C) or Q_w(C).
// ---------------------------------------------------------------------------

//' Compute node impurity Q(C) or Q_w(C)
//'
//' Given class-frequency counts for a node, computes the (weighted) average
//' Gini impurity:
//' \deqn{Q(C) = \frac{1}{M} \sum_{m=1}^{M} \lambda_m \,
//'        \check{p}_m(C)\,(1-\check{p}_m(C))}
//' where \eqn{\check{p}_m(C) = \texttt{counts}[m] / n}.
//' Pass `lambda = rep(1, M)` for the unweighted criterion.
//'
//' @param counts Integer vector of class counts (length M).
//' @param n Total observations in the node.
//' @param M Number of outcome classes.
//' @param lambda Numeric weight vector of length M.
//'
//' @return Scalar node impurity.
//' @keywords internal
//' @export
// [[Rcpp::export]]
double node_impurity_cpp(Rcpp::IntegerVector counts,
                          int                 n,
                          int                 M,
                          Rcpp::NumericVector lambda) {
  return compute_impurity(counts.begin(), n, M, lambda.begin());
}

// ---------------------------------------------------------------------------
// find_best_split_cpp
// Exported for unit testing; considers all k features (mtry = k).
// y is 1-indexed (values 1..M), as passed from R.
//
// Parameters
//   y             : integer vector, values 1..M, length = node size
//   x             : numeric matrix, node_size rows x k columns
//   M             : number of outcome classes
//   lambda        : numeric weight vector, length M
//   min_node_size : minimum observations per child after split
//
// Returns a named List:
//   $feature   : 1-based column index of the best feature (0 if not found)
//   $threshold : split value; observation goes left iff x[j] <= threshold
//   $impurity  : Q(left) + Q(right) at the best split (Inf if not found)
//   $found     : TRUE iff a valid split exists
// ---------------------------------------------------------------------------

//' Find the best split for a node
//'
//' Exhaustively searches all features and candidate thresholds for the split
//' that minimises `Q(left) + Q(right)`.  Ties in feature values are skipped
//' (no split between identical values).
//'
//' @param y Integer vector of class labels (1..M); length = node size.
//' @param x Numeric matrix of covariates for node observations (node_size x k).
//' @param M Number of outcome classes.
//' @param lambda Numeric weight vector of length M.
//' @param min_node_size Minimum number of observations in each child node.
//'
//' @return A named list with elements `feature` (1-based integer), `threshold`
//'   (numeric), `impurity` (numeric), and `found` (logical).
//' @keywords internal
//' @export
// [[Rcpp::export]]
Rcpp::List find_best_split_cpp(Rcpp::IntegerVector y,
                                Rcpp::NumericMatrix x,
                                int                 M,
                                Rcpp::NumericVector lambda,
                                int                 min_node_size) {
  const int n = y.size();
  const int k = x.ncol();

  // Convert 1-indexed y to 0-indexed
  std::vector<int> y0(n);
  for (int i = 0; i < n; ++i) y0[i] = y[i] - 1;

  // Trivial obs array: the node IS the full data passed in
  // (n_boot == n_obs == n, obs = {0, 1, ..., n-1})
  std::vector<int> obs(n);
  std::iota(obs.begin(), obs.end(), 0);

  // All k features
  std::vector<int> all_features(k);
  std::iota(all_features.begin(), all_features.end(), 0);

  // x is column-major (Rcpp::NumericMatrix / Armadillo convention):
  // element (i, j) = x.begin()[j * n + i]
  const double* x_ptr = x.begin();

  std::vector<int>    sort_buf;
  std::vector<double> val_buf;
  const SplitResult sr = find_best_split_internal(
    y0.data(), x_ptr, n,
    obs.data(), n,
    M, lambda.begin(), min_node_size,
    all_features.data(), k,
    sort_buf, val_buf
  );

  return Rcpp::List::create(
    Rcpp::Named("feature")   = sr.feature,
    Rcpp::Named("threshold") = sr.threshold,
    Rcpp::Named("impurity")  = sr.impurity,
    Rcpp::Named("found")     = sr.found
  );
}
