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

#include "jocf_internal.h"
// [[Rcpp::depends(RcppArmadillo)]]

// ---------------------------------------------------------------------------
// find_best_split_internal  (declared in jocf_internal.h)
//
// Searches feature_subset for the split minimising Q(left) + Q(right).
// y is 0-indexed (values 0..M-1).
// ---------------------------------------------------------------------------
SplitResult find_best_split_internal(
  const arma::ivec& y,
  const arma::mat&  x,
  int               M,
  const arma::vec&  lambda,
  int               min_node_size,
  const arma::uvec& feature_subset
) {
  const int n = static_cast<int>(y.n_elem);

  // Total class counts for this node
  arma::ivec total_counts(M, arma::fill::zeros);
  for (int i = 0; i < n; ++i) total_counts[y[i]]++;

  double best_impurity  = arma::datum::inf;
  int    best_feature   = 0;        // 1-based; 0 = not found
  double best_threshold = 0.0;

  arma::ivec left_counts(M), right_counts(M);

  for (arma::uword fi = 0; fi < feature_subset.n_elem; ++fi) {
    const int j = static_cast<int>(feature_subset[fi]);  // 0-based column
    arma::vec xj = x.col(j);
    const arma::uvec ord = arma::sort_index(xj);          // ascending

    left_counts.zeros();
    right_counts = total_counts;
    int n_left = 0, n_right = n;

    for (int ii = 0; ii < n - 1; ++ii) {
      const int obs = static_cast<int>(ord[ii]);
      const int m   = y[obs];    // 0-indexed class

      left_counts[m]++;
      right_counts[m]--;
      ++n_left;
      --n_right;

      // No split between identical feature values
      if (xj[ord[ii]] >= xj[ord[ii + 1]]) continue;

      // Enforce minimum node size on both children
      if (n_left  < min_node_size) continue;
      if (n_right < min_node_size) continue;

      const double imp =
          compute_impurity(left_counts,  n_left,  M, lambda) +
          compute_impurity(right_counts, n_right, M, lambda);

      if (imp < best_impurity) {
        best_impurity  = imp;
        best_feature   = j + 1;    // convert to 1-based
        best_threshold = 0.5 * (xj[ord[ii]] + xj[ord[ii + 1]]);
      }
    }
  }

  return {best_feature, best_threshold, best_impurity, (best_feature > 0)};
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
//' @export
// [[Rcpp::export]]
double node_impurity_cpp(Rcpp::IntegerVector counts,
                          int                 n,
                          int                 M,
                          Rcpp::NumericVector lambda) {
  const arma::ivec cnt = Rcpp::as<arma::ivec>(counts);
  const arma::vec  lam = Rcpp::as<arma::vec>(lambda);
  return compute_impurity(cnt, n, M, lam);
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
  arma::ivec y0(n);
  for (int i = 0; i < n; ++i) y0[i] = y[i] - 1;

  const arma::mat xmat = Rcpp::as<arma::mat>(x);
  const arma::vec lam  = Rcpp::as<arma::vec>(lambda);

  // Consider all k features
  arma::uvec all_features = arma::regspace<arma::uvec>(0, k - 1);

  const SplitResult sr = find_best_split_internal(y0, xmat, M, lam,
                                                   min_node_size, all_features);

  return Rcpp::List::create(
    Rcpp::Named("feature")   = sr.feature,
    Rcpp::Named("threshold") = sr.threshold,
    Rcpp::Named("impurity")  = sr.impurity,
    Rcpp::Named("found")     = sr.found
  );
}
