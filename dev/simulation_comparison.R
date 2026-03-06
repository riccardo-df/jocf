# simulation_comparison.R
#
# Speed and prediction performance: jocf vs ocf
#
# DGP: generate_ordered_data() from the ocf package.
#   Latent Y* = x1*b1 + ... + x6*b6 + eps (logistic noise), discretised into
#   M = 3 ordered classes.  Returns true conditional probabilities P(Y=m|X=x).
#
# Both estimators are run with all available cores.
#
# NOTE on tree counts:
#   ocf(n_trees=B) grows M separate forests of B trees each → M*B total trees.
#   jocf(n_trees=B) grows 1 joint forest of B trees         →   B total trees.
#   Panel A (same parameter): both use B = N_TREES.
#   Panel B (budget-matched): ocf uses B/M trees so total tree count is equal.
#
# Metrics (from ocf package):
#   MSE_true  — MSE vs true P(Y=m|X)     (oracle; only available in simulation)
#   RPS_true  — Ranked Probability Score vs true P(Y=m|X)
#   MSE_obs   — MSE vs observed outcome indicators  (practical, no oracle needed)
#   RPS_obs   — RPS vs observed outcomes
#   CE        — Misclassification rate  (hard label = argmax of predicted probs)

library(jocf)
library(ocf)

# ── Parameters ────────────────────────────────────────────────────────────────
N_TRAIN <- 500
N_TEST  <- 200
N_TOTAL <- N_TRAIN + N_TEST
N_TREES <- 500          # trees for jocf; ocf uses N_TREES or N_TREES/M
M       <- 3            # number of classes (fixed by generate_ordered_data)
MIN_NS  <- 5
N_REP   <- 10
SEED    <- 42

# ── Storage ───────────────────────────────────────────────────────────────────
metric_names <- c("MSE_true", "RPS_true", "MSE_obs", "RPS_obs", "CE")
# 3 estimators: jocf, ocf_same (B=N_TREES), ocf_budgeted (B=N_TREES/M)
est_names <- c("jocf", "ocf_same", "ocf_budget")

res  <- array(NA_real_,
              dim      = c(N_REP, length(est_names), length(metric_names)),
              dimnames = list(NULL, est_names, metric_names))
tims <- matrix(NA_real_, N_REP, length(est_names),
               dimnames = list(NULL, est_names))

ocf_budget_trees <- max(1L, round(N_TREES / M))

cat(sprintf(
  "Setup: n_train=%d  n_test=%d  n_trees=%d  ocf_budget_trees=%d  reps=%d\n\n",
  N_TRAIN, N_TEST, N_TREES, ocf_budget_trees, N_REP))

# ── Helper: collect metrics for a predicted probability matrix ────────────────
collect_metrics <- function(p_hat, Y_te, p_true) {
  c(MSE_true = mean_squared_error(p_true, p_hat, use.true = TRUE),
    RPS_true = mean_ranked_score(p_true,  p_hat, use.true = TRUE),
    MSE_obs  = mean_squared_error(Y_te,   p_hat),
    RPS_obs  = mean_ranked_score(Y_te,    p_hat),
    CE       = classification_error(Y_te, apply(p_hat, 1, which.max)))
}

# ── Simulation loop ───────────────────────────────────────────────────────────
for (r in seq_len(N_REP)) {
  set.seed(SEED + r)

  # --- Data ---
  dat  <- generate_ordered_data(N_TOTAL)
  samp <- dat$sample
  Y_all <- samp$Y
  X_all <- as.matrix(samp[, -1])

  idx_tr <- seq_len(N_TRAIN);  idx_te <- seq(N_TRAIN + 1L, N_TOTAL)
  Y_tr   <- Y_all[idx_tr];     X_tr   <- X_all[idx_tr, ]
  Y_te   <- Y_all[idx_te];     X_te   <- X_all[idx_te, ]
  p_true <- dat$true_probs[idx_te, ]

  k    <- ncol(X_tr)
  mtry <- floor(sqrt(k))

  # --- Fit jocf ---
  tims[r, "jocf"] <- system.time(
    fit_j <- jocf(Y_tr, X_tr,
                  num.trees = N_TREES, min.node.size = MIN_NS,
                  mtry = mtry, num.threads = NULL)
  )["elapsed"]

  # --- Fit ocf (same n_trees parameter) ---
  tims[r, "ocf_same"] <- system.time(
    fit_os <- ocf(Y_tr, X_tr,
                  n.trees = N_TREES, min.node.size = MIN_NS,
                  mtry = mtry, n.threads = 0L)
  )["elapsed"]

  # --- Fit ocf (budget-matched: N_TREES/M total trees) ---
  tims[r, "ocf_budget"] <- system.time(
    fit_ob <- ocf(Y_tr, X_tr,
                  n.trees = ocf_budget_trees, min.node.size = MIN_NS,
                  mtry = mtry, n.threads = 0L)
  )["elapsed"]

  # --- Predict ---
  p_j  <- predict(fit_j, X_te)$probabilities
  p_os <- predict(fit_os, X_te)$probabilities
  p_ob <- predict(fit_ob, X_te)$probabilities

  # --- Metrics ---
  res[r, "jocf",       ] <- collect_metrics(p_j,  Y_te, p_true)
  res[r, "ocf_same",   ] <- collect_metrics(p_os, Y_te, p_true)
  res[r, "ocf_budget", ] <- collect_metrics(p_ob, Y_te, p_true)

  cat(sprintf("Rep %2d/%d  jocf: %4.1fs  ocf(%d): %4.1fs  ocf(%d): %4.1fs\n",
              r, N_REP,
              tims[r,"jocf"],
              N_TREES,       tims[r,"ocf_same"],
              ocf_budget_trees, tims[r,"ocf_budget"]))
}

# ── Summary ───────────────────────────────────────────────────────────────────
hdr <- function(title) {
  cat("\n", strrep("─", 72), "\n", title, "\n", strrep("─", 72), "\n", sep="")
}

hdr("SPEED  (seconds, mean ± sd)")
cat(sprintf("  %-20s %6.2f ± %.2f   [%d total trees]\n",
            "jocf",
            mean(tims[,"jocf"]),   sd(tims[,"jocf"]),
            N_TREES))
cat(sprintf("  %-20s %6.2f ± %.2f   [%d total trees]\n",
            sprintf("ocf  (B=%d)", N_TREES),
            mean(tims[,"ocf_same"]),   sd(tims[,"ocf_same"]),
            M * N_TREES))
cat(sprintf("  %-20s %6.2f ± %.2f   [%d total trees]\n",
            sprintf("ocf  (B=%d)", ocf_budget_trees),
            mean(tims[,"ocf_budget"]), sd(tims[,"ocf_budget"]),
            M * ocf_budget_trees))

hdr("PREDICTION METRICS  (mean ± sd)  —  lower is better")
cat(sprintf("  %-12s  %-22s  %-22s  %-22s\n",
            "Metric",
            sprintf("jocf  (B=%d)", N_TREES),
            sprintf("ocf   (B=%d)", N_TREES),
            sprintf("ocf   (B=%d)", ocf_budget_trees)))
cat(strrep("─", 72), "\n")
for (m in metric_names) {
  vals <- sapply(est_names, function(e) res[, e, m])
  cat(sprintf("  %-12s  %6.5f ± %.5f  %6.5f ± %.5f  %6.5f ± %.5f\n",
              m,
              mean(vals[,"jocf"]),       sd(vals[,"jocf"]),
              mean(vals[,"ocf_same"]),   sd(vals[,"ocf_same"]),
              mean(vals[,"ocf_budget"]), sd(vals[,"ocf_budget"])))
}
cat(strrep("─", 72), "\n")
cat("*_true: oracle (known DGP probabilities).  *_obs: empirical.\n")
cat(sprintf("Budget-matched: jocf(%d) vs ocf(%d) use the same total tree count (%d).\n",
            N_TREES, ocf_budget_trees, N_TREES))
