"""
Extension: Rao-Blackwellized estimate of the success probability (identity design).

In the identity design the off-support columns X_j (j not in S) are i.i.d. N(0, I_n) and independent
of (X_S, w). Given X_S and w, the restricted solution and the residual r = y - X_S beta_check_S are
fixed, so z_j = X_j^T r / (n lam) are i.i.d. N(0, ||r||^2 / (n lam)^2), and

    P(success | X_S, w) = 1{signs correct} * (2 Phi(n lam / ||r||) - 1)^(p - k).

Averaging this conditional probability instead of the 0/1 success indicator gives an unbiased
estimate with a smaller variance (Rao-Blackwell). Both are computed on the same replicates, with the
same seeds as the main study.

Run from the project root:
    python -m src.rao_blackwell                 # identity design, all regimes, p and theta; 200 reps
    python -m src.rao_blackwell --n-reps 50
"""

import argparse
import copy
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import special          # used as special.ndtr: a bare ndtr cannot be sent to joblib workers
from scipy.linalg import cho_factor, cho_solve

from src.dgps import generate_replicate
from src.figures import INK, MUTED, P_STYLE, save_figure, set_style
from src.simulation import CONFIG, build_conditions, replicate_rng

logger = logging.getLogger(__name__)


def log_prob_all_inside(t, m):
    """log P(m i.i.d. |N(0, 1)| are all below t) = m * log(1 - 2 Phi(-t)).

    log1p(-2 Phi(-t)) stays accurate when Phi(-t) is tiny (large t), where 1 - 2 Phi(-t) would round to 1.
    """
    return m * np.log1p(-2.0 * special.ndtr(-t))


def replicate_estimates(data, lam):
    """(indicator, rao_blackwell) for one replicate: observed success, and P(success | X_S, w)."""
    X, y, beta, support = data["X"], data["y"], data["beta"], data["support"]
    n, p = X.shape
    k = len(support)
    if n <= k:
        return 0.0, 0.0

    X_S = X[:, support]
    z_S = np.sign(beta[support])
    beta_check_S = cho_solve(cho_factor(X_S.T @ X_S), X_S.T @ y - n * lam * z_S)
    if np.min(z_S * beta_check_S) <= 0.0:          # a sign is wrong: failure, whatever X_{S^c} is
        return 0.0, 0.0

    residual = y - X_S @ beta_check_S
    off_support = np.ones(p, dtype=bool)
    off_support[support] = False
    indicator = float(np.max(np.abs((X.T @ residual)[off_support])) / (n * lam) < 1.0)
    rao_blackwell = float(np.exp(log_prob_all_inside(n * lam / np.linalg.norm(residual), p - k)))
    return indicator, rao_blackwell


def run_condition(cond, config):
    """Both estimates and their Monte Carlo standard errors for one condition."""
    estimates = []
    for rep in range(config["n_reps"]):
        rng = replicate_rng(config["master_seed"], cond["design"], cond["regime"], cond["p"], cond["theta"], rep)
        data = generate_replicate(cond["n"], cond["p"], cond["k"], config["sigma"], config["beta_min"], rng,
                                  design="identity")
        estimates.append(replicate_estimates(data, cond["lam"]))
    est = np.array(estimates)
    n_reps = len(est)
    row = {key: cond[key] for key in ("regime", "p", "k", "theta", "n")}
    row.update({"n_reps": n_reps,
                "rate_indicator": est[:, 0].mean(), "rate_rb": est[:, 1].mean(),
                "se_indicator": est[:, 0].std(ddof=1) / np.sqrt(n_reps),
                "se_rb": est[:, 1].std(ddof=1) / np.sqrt(n_reps)})
    return row


def fig_rao_blackwell(df, out_stem, regime="fractional"):
    """Left: both estimates vs theta (one regime). Right: variance ratio vs theta."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for p, style in P_STYLE.items():
        g = df[(df["regime"] == regime) & (df["p"] == p)].sort_values("theta")
        ax1.errorbar(g["theta"], g["rate_indicator"], yerr=1.96 * g["se_indicator"], fmt="o", markersize=3,
                     mfc="none", color=style["color"], alpha=0.7, label=f"p = {p}, indicator")
        ax1.plot(g["theta"], g["rate_rb"], color=style["color"], label=f"p = {p}, Rao-Blackwell")
        ax1.fill_between(g["theta"], g["rate_rb"] - 1.96 * g["se_rb"], g["rate_rb"] + 1.96 * g["se_rb"],
                         color=style["color"], alpha=0.2, linewidth=0)
        ok = g["var_ratio"].notna() & (g["rate_rb"] > 0.05)    # skip the noisy region where success is rare
        ax2.plot(g.loc[ok, "theta"], g.loc[ok, "var_ratio"], marker=style["marker"], markersize=4,
                 color=style["color"], label=f"p = {p}")
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel("estimated success probability")
    ax1.set_title(f"Identity design, {regime} regime (95% intervals)", color=INK)
    ax1.legend(fontsize=7, ncol=2)
    ax2.axhline(1.0, linestyle="--", color=MUTED)
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel("variance ratio (indicator / Rao-Blackwell)")
    ax2.set_title("How many times fewer reps Rao-Blackwell needs", color=INK)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_stem)


def parse_args():
    parser = argparse.ArgumentParser(description="Rao-Blackwellized success probability (identity design).")
    parser.add_argument("--n-reps", type=int, default=200, help="replicates per condition")
    parser.add_argument("--n-jobs", type=int, default=-1, help="worker processes (-1: all cores)")
    parser.add_argument("--out", default="results/extensions/rao_blackwell.csv")
    parser.add_argument("--fig", default="results/figures/ext_rao_blackwell")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = copy.deepcopy(CONFIG)
    config.update(designs=["identity"], n_reps=args.n_reps)

    rows = Parallel(n_jobs=args.n_jobs)(delayed(run_condition)(cond, config) for cond in build_conditions(config))
    df = pd.DataFrame(rows)
    both_vary = (df["se_indicator"] > 0) & (df["se_rb"] > 0)     # ratio is meaningless if a variance is 0
    df["var_ratio"] = np.where(both_vary, (df["se_indicator"] / df["se_rb"]) ** 2, np.nan)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Saved %s", out)

    set_style()
    fig_rao_blackwell(df, Path(args.fig))

    mid = df[(df["rate_rb"] > 0.05) & (df["rate_rb"] < 0.95)]      # transition region
    print(f"\nConditions in the transition region (0.05 < rate < 0.95): {len(mid)}")
    print("Variance ratio there: median %.1f, min %.1f, max %.1f"
          % (mid["var_ratio"].median(), mid["var_ratio"].min(), mid["var_ratio"].max()))
    print("Largest |indicator - RB| / se_indicator there: %.2f"
          % (np.abs(mid["rate_indicator"] - mid["rate_rb"]) / mid["se_indicator"]).max())


if __name__ == "__main__":
    main()
