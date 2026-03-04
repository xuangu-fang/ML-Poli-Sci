"""
simulate_tca.py
===============
Monte Carlo Simulation Study for:
  "Transfer Component Analysis for Domain Adaptation in Political Science"
  (SMR paper – Section 5)

Reproduces Tables 1–4 and Figures 1–3 from the simulation study.

Data-Generating Process (Section 5.1)
--------------------------------------
  • p = 20 features, AR(1) covariance (ρ = 0.3)
  • β* = (1.5, −1.2, 1.0, −0.8, 0.7, −0.6, 0.5, −0.4, 0, …, 0)
  • Feature groups:
      – Causal  (dims  1– 8): stable logistic signal; no domain shift
      – Mixed   (dims  9–12): zero coefficient; no spurious/shift signal
      – Proxy   (dims 13–20): spuriously predictive in source; shifted in target
  • Outcome: P(Y=1|X) = σ(X β*)   (true signal from causal features only)
  • Source proxy augmentation: X_proxy += δ × z_s_std
      (creates within-source predictive value that does not transfer)
  • Target shift: μ_target = δ × shift_dir × 2.5  (shift in proxy subspace)
  • δ ∈ {0.0, 0.5, 1.0, 1.5};  n_s ∈ {500, 1000};  n_t = 0.6 × n_s
  • B = 200 Monte Carlo replications per condition (1 600 total)

Methods (Section 5.2)
----------------------
  1. Naive Logistic Regression   – unpenalised, no adaptation
  2. Elastic-Net                 – λ=0.05, α=0.5; no adaptation
  3. IPW                         – propensity via elastic-net; reweighted EN
  4. TCA + Elastic-Net           – k=10 components, μ=1.0; Pan et al. (2011)

Metrics (Section 5.3)
----------------------
  • RMSE  – root mean squared error vs. true target probabilities P(Y=1|X_t)
  • Bias  – mean(predicted − true) on target domain
  • MMD   – linear-kernel MMD before/after TCA projection

Outputs
-------
  SMR/simulation_results.csv      raw condition-level results
  SMR/table1_rmse.csv             Table 1 (RMSE)
  SMR/table2_bias.csv             Table 2 (bias, n_s=1000)
  SMR/table3_mmd.csv              Table 3 (MMD reduction)
  SMR/table4_relative.csv         Table 4 (relative improvement TCA vs. EN)
  SMR/fig1_rmse_by_shift.png      Figure 1
  SMR/fig2_mmd_reduction.png      Figure 2
  SMR/fig3_bias.png               Figure 3

Usage
-----
  python simulate_tca.py

Requirements
------------
  numpy, pandas, matplotlib   (standard scientific Python stack)
  scipy, scikit-learn         (optional; see note below)

Optional sklearn/scipy note
---------------------------
  This script is fully self-contained using numpy-only implementations
  of logistic regression (gradient descent + FISTA) and TCA (Cholesky
  generalised eigenvalue solver).  If scipy and scikit-learn are
  available in your environment, you can replace the calls to
  fit_logistic() / fit_elasticnet() with sklearn's LogisticRegression
  (penalty='elasticnet', solver='saga', C=1/EN_LAM, l1_ratio=EN_ALPHA)
  and from scipy.linalg import eigh in the LinearTCA class for faster
  and potentially more precise replication of the paper's exact numbers.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Output directory (same folder as this script)
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Simulation parameters
# ─────────────────────────────────────────────────────────────────────────────
P         = 20           # total features
P_CAUSAL  = 8            # indices 0–7  (causal)
P_MIXED   = 4            # indices 8–11 (mixed; zero coefficients)
P_PROXY   = 8            # indices 12–19 (proxy / spurious)
RHO       = 0.3          # AR(1) correlation

# True coefficient vector β* — sparse on causal features
BETA = np.array(
    [1.5, -1.2, 1.0, -0.8, 0.7, -0.6, 0.5, -0.4]   # causal (dims 1–8)
    + [0.0] * 12                                        # mixed + proxy
)

DELTAS  = [0.0, 0.5, 1.0, 1.5]   # shift severity δ
NS_LIST = [500, 1000]             # source sample sizes n_s
B       = 200                     # Monte Carlo replications per condition
SEED    = 42                      # master random seed (for reproducibility)

# Elastic-Net hyperparameters
#   sklearn convention: penalty = lam * [alpha * ||β||₁ + (1-alpha)/2 * ||β||₂²]
EN_LAM   = 0.05   # regularisation strength λ
EN_ALPHA = 0.5    # l1_ratio α  (0 = ridge, 1 = lasso)

# TCA hyperparameters
TCA_K  = 10    # number of transfer components
TCA_MU = 1.0   # regularisation weight μ


# ─────────────────────────────────────────────────────────────────────────────
# Math utilities
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def make_ar1_cov(p: int, rho: float) -> np.ndarray:
    """AR(1) covariance Σ_{ij} = ρ^|i−j|."""
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def standardise(X_train: np.ndarray, X_test: np.ndarray):
    """Z-score normalise: fit on X_train, apply to both."""
    mu  = X_train.mean(axis=0)
    sig = X_train.std(axis=0) + 1e-10
    return (X_train - mu) / sig, (X_test - mu) / sig


def linear_mmd(X_s: np.ndarray, X_t: np.ndarray) -> float:
    """Linear-kernel MMD²: ‖μ_s − μ_t‖²."""
    diff = X_s.mean(axis=0) - X_t.mean(axis=0)
    return float(diff @ diff)


# ─────────────────────────────────────────────────────────────────────────────
# Logistic regression estimators (numpy-only)
# ─────────────────────────────────────────────────────────────────────────────

def _bce_loss_and_grad(beta, X_aug, y, w):
    """
    Weighted binary cross-entropy and gradient.

    Parameters
    ----------
    beta  : (p+1,) coefficients [β; b]  (b = intercept)
    X_aug : (n, p+1) feature matrix with intercept column appended
    y     : (n,) binary labels
    w     : (n,) normalised sample weights (sum to 1)
    """
    Xb    = X_aug @ beta
    p_hat = sigmoid(Xb)
    # Numerically stable BCE: log(1+exp(x)) = max(x,0) + log(1+exp(-|x|))
    loss  = np.sum(w * (np.maximum(Xb, 0) + np.log1p(np.exp(-np.abs(Xb))) - y * Xb))
    grad  = X_aug.T @ (w * (p_hat - y))
    return loss, grad


def fit_logistic(X: np.ndarray, y: np.ndarray,
                 max_iter: int = 500, tol: float = 1e-5) -> np.ndarray:
    """
    Unpenalised logistic regression via gradient descent with backtracking
    line search (Armijo condition).

    Returns β_aug = [β; b] ∈ ℝ^{p+1} (last element is intercept).
    """
    n, p  = X.shape
    X_aug = np.column_stack([X, np.ones(n)])
    beta  = np.zeros(p + 1)
    w     = np.full(n, 1.0 / n)

    for _ in range(max_iter):
        loss, grad = _bce_loss_and_grad(beta, X_aug, y, w)
        gnorm = np.linalg.norm(grad)
        if gnorm < tol:
            break
        # Backtracking line search
        step = 1.0
        slope = np.dot(grad, grad)
        for _ in range(40):
            beta_new = beta - step * grad
            loss_new, _ = _bce_loss_and_grad(beta_new, X_aug, y, w)
            if loss_new <= loss - 0.5 * step * slope:
                break
            step *= 0.5
        beta = beta - step * grad

    return beta


def fit_elasticnet(X: np.ndarray, y: np.ndarray,
                   lam: float = EN_LAM, alpha: float = EN_ALPHA,
                   sample_weight: np.ndarray = None,
                   max_iter: int = 1000, tol: float = 1e-5) -> np.ndarray:
    """
    Elastic-net penalised logistic regression via FISTA (proximal gradient).

    Objective:
        min_{β,b} (1/n) Σ_i w_i * BCE(y_i, σ(x_iᵀβ + b))
                  + λ [ α ‖β‖₁ + (1−α)/2 ‖β‖₂² ]

    The intercept b (last element) is not penalised.

    Returns β_aug = [β; b] ∈ ℝ^{p+1}.
    """
    n, p  = X.shape
    X_aug = np.column_stack([X, np.ones(n)])

    # Normalised sample weights
    if sample_weight is None:
        w = np.full(n, 1.0 / n)
    else:
        w = sample_weight / sample_weight.sum()

    # Lipschitz constant of gradient of smooth part
    #   L(f) = λ_max(X_aug^T diag(w) X_aug) / 4 + λ(1−α)
    #   Cheaper upper bound: max_i( w_i * ||x_i||² ) / 4 + λ(1−α)
    XTwX  = (X_aug * w[:, None]).T @ X_aug       # (p+1, p+1) weighted Gram matrix
    L_lip = np.linalg.eigvalsh(XTwX)[-1] / 4.0 + lam * (1 - alpha)
    step  = 1.0 / (L_lip + 1e-10)

    beta      = np.zeros(p + 1)
    beta_prev = beta.copy()
    t_prev    = 1.0
    thresh    = step * lam * alpha   # soft-threshold level for L1

    for it in range(max_iter):
        # FISTA momentum update
        t      = (1.0 + np.sqrt(1.0 + 4.0 * t_prev ** 2)) / 2.0
        y_mom  = beta + (t_prev - 1.0) / t * (beta - beta_prev)
        beta_prev = beta.copy()
        t_prev    = t

        # Gradient of smooth part at y_mom
        p_hat  = sigmoid(X_aug @ y_mom)
        grad   = X_aug.T @ (w * (p_hat - y))
        grad[:-1] += lam * (1.0 - alpha) * y_mom[:-1]  # L2 term (not intercept)

        # Gradient step
        z = y_mom - step * grad

        # Proximal operator: soft-thresholding on β (not intercept)
        beta = np.sign(z) * np.maximum(np.abs(z) - thresh, 0.0)
        beta[-1] = z[-1]                                # intercept unchanged

        if np.max(np.abs(beta - beta_prev)) < tol:
            break

    return beta


def predict_proba(beta_aug: np.ndarray, X: np.ndarray) -> np.ndarray:
    """P(Y=1|X) from augmented coefficient vector [β; b]."""
    n = X.shape[0]
    X_aug = np.column_stack([X, np.ones(n)])
    return sigmoid(X_aug @ beta_aug)


# ─────────────────────────────────────────────────────────────────────────────
# Transfer Component Analysis  (linear kernel; Pan et al. 2011)
# ─────────────────────────────────────────────────────────────────────────────

class LinearTCA:
    """
    Linear Transfer Component Analysis.

    Finds a k-dimensional projection W ∈ ℝ^{p × k} that minimises the
    MMD between projected source and target distributions while preserving
    variance in the data.

    Generalised eigenvalue problem (linear kernel  K = X Xᵀ):

        KHK w = λ (KLK + μI) w

    where  H  is the centering matrix and  L  encodes source/target membership.
    The primal projection is  W = Xᵀ W_dual  ∈ ℝ^{p × k}.

    Reference
    ---------
    Pan, S. J., Tsang, I. W., Kwok, J. T., & Yang, Q. (2011).
    Domain Adaptation via Transfer Component Analysis.
    IEEE Transactions on Neural Networks, 22(2), 199–210.
    """

    def __init__(self, k: int = 10, mu: float = 1.0):
        self.k = k
        self.mu = mu
        self.W_primal_ = None   # (p, k) projection matrix

    def fit(self, X_s: np.ndarray, X_t: np.ndarray) -> "LinearTCA":
        n_s, p = X_s.shape
        n_t    = X_t.shape[0]
        n      = n_s + n_t
        X      = np.vstack([X_s, X_t])          # (n, p)

        # Linear kernel  K ∈ ℝ^{n × n}
        K = X @ X.T

        # MMD matrix L
        L                = np.zeros((n, n))
        L[:n_s, :n_s]    =  1.0 / (n_s * n_s)
        L[n_s:, n_s:]    =  1.0 / (n_t * n_t)
        L[:n_s, n_s:]    = -1.0 / (n_s * n_t)
        L[n_s:, :n_s]    = -1.0 / (n_s * n_t)

        # Centering matrix  H = I − (1/n) 11ᵀ
        H = np.eye(n) - np.ones((n, n)) / n

        # Build matrices for generalised eigenvalue problem
        #   A (denominator): KLK + μI
        #   B (numerator):   KHK
        A = K @ L @ K + self.mu * np.eye(n)
        B = K @ H @ K

        # Solve  B w = λ A w  via Cholesky reduction to standard problem
        #   A = L_A L_Aᵀ  →  (L_A⁻¹ B L_A⁻ᵀ) v = λ v,  w = L_A⁻ᵀ v
        try:
            L_A  = np.linalg.cholesky(A + 1e-8 * np.eye(n))   # stabilise
            L_Ai = np.linalg.solve(L_A, np.eye(n))             # L_A⁻¹
            C    = L_Ai @ B @ L_Ai.T                           # symmetric
            C    = (C + C.T) / 2.0                             # enforce symmetry
            eigvals, eigvecs = np.linalg.eigh(C)               # ascending order
            # Back-transform to original eigenvectors
            W_dual = L_Ai.T @ eigvecs[:, -self.k:]             # (n, k)
        except np.linalg.LinAlgError:
            # Fallback: direct eigh on A⁻¹B
            A_reg = A + 1e-6 * np.eye(n)
            eigvals, eigvecs = np.linalg.eigh(
                np.linalg.solve(A_reg, B)
            )
            W_dual = eigvecs[:, -self.k:]                      # (n, k)

        # Primal projection  W = Xᵀ W_dual  ∈ ℝ^{p × k}
        self.W_primal_ = X.T @ W_dual
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X ∈ ℝ^{m × p} → Z ∈ ℝ^{m × k}."""
        if self.W_primal_ is None:
            raise RuntimeError("Call fit() before transform().")
        return X @ self.W_primal_


# ─────────────────────────────────────────────────────────────────────────────
# Data-generating process  (Section 5.1)
# ─────────────────────────────────────────────────────────────────────────────

def generate_data(n_s: int, n_t: int, delta: float,
                  rng: np.random.Generator):
    """
    One draw from the Section-5.1 DGP.

    Returns
    -------
    X_s  : (n_s, p)  source features (with proxy augmentation)
    y_s  : (n_s,)    source binary outcomes
    X_t  : (n_t, p)  target features (shifted proxy means)
    p_t  : (n_t,)    true target probabilities P(Y=1|X_t)
    """
    cov = make_ar1_cov(P, RHO)

    # ── Source domain ────────────────────────────────────────────────────────
    X_s = rng.multivariate_normal(np.zeros(P), cov, size=n_s)

    # Standardised causal linear predictor
    z_s     = X_s @ BETA
    z_s_std = (z_s - z_s.mean()) / (z_s.std() + 1e-10)

    # Spurious augmentation of proxy features in source
    # (mirrors political self-selection among observed voters)
    X_s[:, 12:] += delta * z_s_std[:, None]

    # Binary outcomes (logistic model on causal signal only)
    p_s = sigmoid(X_s @ BETA)
    y_s = rng.binomial(1, p_s)

    # ── Target domain ────────────────────────────────────────────────────────
    shift_dir       = np.zeros(P)
    shift_dir[12:]  = 1.0 / np.sqrt(P_PROXY)       # unit vector in proxy subspace
    mu_target       = delta * shift_dir * 2.5

    X_t = rng.multivariate_normal(mu_target, cov, size=n_t)
    p_t = sigmoid(X_t @ BETA)                       # no spurious correlation

    return X_s, y_s, X_t, p_t


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation() -> pd.DataFrame:
    rng     = np.random.default_rng(SEED)
    records = []

    n_total = len(DELTAS) * len(NS_LIST) * B
    done    = 0
    print(f"Running {B} reps × {len(DELTAS) * len(NS_LIST)} conditions "
          f"= {n_total} replications …\n")

    for delta in DELTAS:
        for n_s in NS_LIST:
            n_t = int(0.6 * n_s)

            # Accumulation arrays
            rmse_naive = np.empty(B);  bias_naive = np.empty(B)
            rmse_en    = np.empty(B);  bias_en    = np.empty(B)
            rmse_ipw   = np.empty(B);  bias_ipw   = np.empty(B)
            rmse_tca   = np.empty(B);  bias_tca   = np.empty(B)
            mmd_before = np.empty(B);  mmd_after  = np.empty(B)

            for b in range(B):
                # ── Data ──────────────────────────────────────────────────
                X_s, y_s, X_t, p_t = generate_data(n_s, n_t, delta, rng)

                # Z-score standardise (source fit; applied to target)
                X_s_sc, X_t_sc = standardise(X_s, X_t)

                # ── MMD before TCA ─────────────────────────────────────────
                mmd_before[b] = linear_mmd(X_s_sc, X_t_sc)

                # ══ Method 1: Naive Logistic Regression ════════════════════
                beta_naive = fit_logistic(X_s_sc, y_s)
                p_hat      = predict_proba(beta_naive, X_t_sc)
                rmse_naive[b] = np.sqrt(np.mean((p_hat - p_t) ** 2))
                bias_naive[b] = np.mean(p_hat - p_t)

                # ══ Method 2: Elastic-Net (no adaptation) ══════════════════
                beta_en = fit_elasticnet(X_s_sc, y_s, lam=EN_LAM, alpha=EN_ALPHA)
                p_hat   = predict_proba(beta_en, X_t_sc)
                rmse_en[b] = np.sqrt(np.mean((p_hat - p_t) ** 2))
                bias_en[b] = np.mean(p_hat - p_t)

                # ══ Method 3: IPW ═══════════════════════════════════════════
                # Estimate P(domain=target | X) via elastic-net
                X_all = np.vstack([X_s_sc, X_t_sc])
                d_all = np.array([0] * n_s + [1] * n_t, dtype=float)
                beta_prop = fit_elasticnet(X_all, d_all,
                                           lam=EN_LAM, alpha=EN_ALPHA)
                prop_s    = predict_proba(beta_prop, X_s_sc)     # P(target|X_s)

                # Importance weights: ratio of target to source density
                weights = prop_s / (1.0 - prop_s + 1e-10)
                weights = weights / (weights.mean() + 1e-10)     # normalise

                beta_ipw = fit_elasticnet(X_s_sc, y_s, lam=EN_LAM, alpha=EN_ALPHA,
                                          sample_weight=weights)
                p_hat    = predict_proba(beta_ipw, X_t_sc)
                rmse_ipw[b] = np.sqrt(np.mean((p_hat - p_t) ** 2))
                bias_ipw[b] = np.mean(p_hat - p_t)

                # ══ Method 4: TCA + Elastic-Net ═════════════════════════════
                tca = LinearTCA(k=TCA_K, mu=TCA_MU)
                tca.fit(X_s_sc, X_t_sc)

                Z_s = tca.transform(X_s_sc)         # (n_s, k)
                Z_t = tca.transform(X_t_sc)         # (n_t, k)

                # MMD measured in TCA projected subspace (before re-scaling)
                mmd_after[b] = linear_mmd(Z_s, Z_t)

                # Re-standardise projected features for elastic-net fitting
                Z_s_sc, Z_t_sc = standardise(Z_s, Z_t)

                beta_tca = fit_elasticnet(Z_s_sc, y_s, lam=EN_LAM, alpha=EN_ALPHA)
                p_hat    = predict_proba(beta_tca, Z_t_sc)
                rmse_tca[b] = np.sqrt(np.mean((p_hat - p_t) ** 2))
                bias_tca[b] = np.mean(p_hat - p_t)

                done += 1
                if done % 200 == 0:
                    print(f"  {done}/{n_total} replications complete …")

            # ── Aggregate across B replications ───────────────────────────
            se = lambda a: a.std() / np.sqrt(B)
            records.append(dict(
                delta         = delta,
                n_s           = n_s,
                n_t           = n_t,
                rmse_naive    = rmse_naive.mean(),
                se_rmse_naive = se(rmse_naive),
                rmse_en       = rmse_en.mean(),
                se_rmse_en    = se(rmse_en),
                rmse_ipw      = rmse_ipw.mean(),
                se_rmse_ipw   = se(rmse_ipw),
                rmse_tca      = rmse_tca.mean(),
                se_rmse_tca   = se(rmse_tca),
                bias_naive    = bias_naive.mean(),
                se_bias_naive = se(bias_naive),
                bias_en       = bias_en.mean(),
                se_bias_en    = se(bias_en),
                bias_ipw      = bias_ipw.mean(),
                se_bias_ipw   = se(bias_ipw),
                bias_tca      = bias_tca.mean(),
                se_bias_tca   = se(bias_tca),
                mmd_before    = mmd_before.mean(),
                mmd_after     = mmd_after.mean(),
            ))

            r = records[-1]
            print(f"  δ={delta:.1f}  n_s={n_s:4d}  │  "
                  f"RMSE  Naive={r['rmse_naive']:.4f}  "
                  f"EN={r['rmse_en']:.4f}  "
                  f"IPW={r['rmse_ipw']:.4f}  "
                  f"TCA={r['rmse_tca']:.4f}  │  "
                  f"MMD {r['mmd_before']:.4f}→{r['mmd_after']:.4f}")

    print()
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Table builders
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v, se):
    return f"{v:.4f} ({se:.4f})"

def _fmt_b(v, se):
    s = "+" if v >= 0 else ""
    return f"{s}{v:.4f} ({se:.4f})"


def build_table1(df):
    """Table 1 – RMSE by method and condition."""
    rows = []
    for _, r in df.iterrows():
        rows.append({"δ": r.delta, "n_s": int(r.n_s), "n_t": int(r.n_t),
            "Naive Logistic"   : _fmt(r.rmse_naive, r.se_rmse_naive),
            "Elastic-Net"      : _fmt(r.rmse_en,    r.se_rmse_en),
            "IPW"              : _fmt(r.rmse_ipw,   r.se_rmse_ipw),
            "TCA + Elastic-Net": _fmt(r.rmse_tca,   r.se_rmse_tca),
        })
    return pd.DataFrame(rows)


def build_table2(df):
    """Table 2 – Prediction bias (n_s = 1000 only)."""
    rows = []
    for _, r in df[df.n_s == 1000].iterrows():
        rows.append({"δ": r.delta,
            "Naive Logistic"   : _fmt_b(r.bias_naive, r.se_bias_naive),
            "Elastic-Net"      : _fmt_b(r.bias_en,    r.se_bias_en),
            "IPW"              : _fmt_b(r.bias_ipw,   r.se_bias_ipw),
            "TCA + Elastic-Net": _fmt_b(r.bias_tca,   r.se_bias_tca),
        })
    return pd.DataFrame(rows)


def build_table3(df):
    """Table 3 – MMD before/after TCA alignment."""
    rows = []
    for _, r in df.iterrows():
        pct = 100.0 * (r.mmd_before - r.mmd_after) / (r.mmd_before + 1e-12)
        rows.append({"δ": r.delta, "n_s": int(r.n_s),
            "MMD Before TCA": f"{r.mmd_before:.4f}",
            "MMD After TCA" : f"{r.mmd_after:.4f}",
            "Reduction (%)" : f"{pct:.1f}%",
        })
    return pd.DataFrame(rows)


def build_table4(df):
    """Table 4 – Relative improvement TCA+EN vs. Elastic-Net."""
    rows = []
    for _, r in df.iterrows():
        rmse_chg = 100.0 * (r.rmse_en - r.rmse_tca) / (r.rmse_en + 1e-12)
        en_ab    = abs(r.bias_en)
        tca_ab   = abs(r.bias_tca)
        bias_chg = 100.0 * (en_ab - tca_ab) / (en_ab + 1e-12) if en_ab > 1e-6 else float("nan")
        rows.append({"δ": r.delta, "n_s": int(r.n_s),
            "TCA RMSE"        : f"{r.rmse_tca:.4f}",
            "EN RMSE"         : f"{r.rmse_en:.4f}",
            "RMSE Change (%)" : f"{rmse_chg:+.1f}%",
            "TCA |Bias|"      : f"{tca_ab:.4f}",
            "EN |Bias|"       : f"{en_ab:.4f}",
            "Bias Change (%)" : f"{bias_chg:+.1f}%" if not np.isnan(bias_chg) else "n/a",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

LABELS  = ["Naive Logistic", "Elastic-Net", "IPW", "TCA + Elastic-Net"]
COLORS  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
STYLES  = ["--",       "--",      "-.",      "-"]
MARKERS = ["o",        "s",       "^",       "D"]


def _ax_settings(ax, title, xlabel="Shift severity (δ)", ylabel=""):
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(DELTAS)
    ax.grid(True, alpha=0.3, linewidth=0.6)


def figure1_rmse(df, out_path):
    """Figure 1 – RMSE by method and covariate shift severity."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    fig.suptitle(
        "Figure 1. RMSE in Target Domain by Method and Covariate Shift Severity"
        "\n(B = 200 replications)",
        fontsize=10, y=1.02,
    )
    cols    = ["rmse_naive", "rmse_en",    "rmse_ipw",   "rmse_tca"]
    se_cols = ["se_rmse_naive","se_rmse_en","se_rmse_ipw","se_rmse_tca"]
    for ax, n_s in zip(axes, NS_LIST):
        sub = df[df.n_s == n_s]
        for col, sec, lbl, clr, ls, mk in zip(cols, se_cols, LABELS, COLORS, STYLES, MARKERS):
            ax.errorbar(DELTAS, sub[col].values, yerr=1.96 * sub[sec].values,
                        label=lbl, color=clr, ls=ls, marker=mk,
                        markersize=6, linewidth=1.8, capsize=3, elinewidth=0.8)
        _ax_settings(ax, f"$n_s$ = {n_s}", ylabel="RMSE" if n_s == 500 else "")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


def figure2_mmd(df, out_path):
    """Figure 2 – MMD before and after TCA alignment."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    fig.suptitle(
        "Figure 2. MMD Before and After TCA Alignment Across Shift Levels"
        "\n(Mean over B = 200 replications)",
        fontsize=10, y=1.02,
    )
    for ax, n_s in zip(axes, NS_LIST):
        sub = df[df.n_s == n_s]
        ax.plot(DELTAS, sub["mmd_before"].values,
                color="#4C72B0", marker="o", ls="--",
                linewidth=1.8, markersize=6, label="Before TCA")
        ax.plot(DELTAS, sub["mmd_after"].values,
                color="#C44E52", marker="D", ls="-",
                linewidth=1.8, markersize=6, label="After TCA")
        _ax_settings(ax, f"$n_s$ = {n_s}",
                     ylabel="Linear-kernel MMD" if n_s == 500 else "")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


def figure3_bias(df, out_path):
    """Figure 3 – Mean prediction bias by method and shift severity."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    fig.suptitle(
        "Figure 3. Mean Prediction Bias in Target Domain by Method and Shift Severity",
        fontsize=10, y=1.02,
    )
    cols    = ["bias_naive", "bias_en",    "bias_ipw",   "bias_tca"]
    se_cols = ["se_bias_naive","se_bias_en","se_bias_ipw","se_bias_tca"]
    for ax, n_s in zip(axes, NS_LIST):
        sub = df[df.n_s == n_s]
        for col, sec, lbl, clr, ls, mk in zip(cols, se_cols, LABELS, COLORS, STYLES, MARKERS):
            ax.errorbar(DELTAS, sub[col].values, yerr=1.96 * sub[sec].values,
                        label=lbl, color=clr, ls=ls, marker=mk,
                        markersize=6, linewidth=1.8, capsize=3, elinewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.8, ls=":")
        _ax_settings(ax, f"$n_s$ = {n_s}",
                     ylabel="Mean Bias (predicted − true)" if n_s == 500 else "")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("TCA Simulation Study — Section 5 Monte Carlo Experiment")
    print(f"  p={P}, ρ={RHO}, B={B} replications, seed={SEED}")
    print(f"  δ ∈ {DELTAS}")
    print(f"  n_s ∈ {NS_LIST},  n_t = 0.6 × n_s")
    print(f"  EN: λ={EN_LAM}, α={EN_ALPHA}   TCA: k={TCA_K}, μ={TCA_MU}")
    print("=" * 70)
    print()

    # ── Run Monte Carlo ───────────────────────────────────────────────────
    results = run_simulation()

    # ── Save raw results ──────────────────────────────────────────────────
    results.to_csv(os.path.join(OUT_DIR, "simulation_results.csv"), index=False)
    print(f"Raw results → simulation_results.csv")
    print()

    # ── Formatted tables ──────────────────────────────────────────────────
    for name, builder, fname in [
        ("Table 1 – RMSE",                         build_table1, "table1_rmse.csv"),
        ("Table 2 – Bias (n_s=1000)",               build_table2, "table2_bias.csv"),
        ("Table 3 – MMD Reduction",                 build_table3, "table3_mmd.csv"),
        ("Table 4 – Relative Improvement (TCA/EN)", build_table4, "table4_relative.csv"),
    ]:
        tbl = builder(results)
        tbl.to_csv(os.path.join(OUT_DIR, fname), index=False)
        print(f"\n{name}:")
        print(tbl.to_string(index=False))

    # ── Figures ───────────────────────────────────────────────────────────
    print("\nGenerating figures …")
    figure1_rmse(results, os.path.join(OUT_DIR, "fig1_rmse_by_shift.png"))
    figure2_mmd (results, os.path.join(OUT_DIR, "fig2_mmd_reduction.png"))
    figure3_bias(results, os.path.join(OUT_DIR, "fig3_bias.png"))

    print()
    print("=" * 70)
    print("Simulation complete.  All outputs saved to:", OUT_DIR)
    print("=" * 70)
