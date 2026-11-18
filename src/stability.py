"""
Numerical stability check: warnings, exceptions, solver convergence, conditioning of X_S
and near-ties, for every condition.

Run from the project root:
    python -m src.stability                  # all 864 conditions, 10 reps each
    python -m src.stability --n-reps 20
"""

import argparse
import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.dgps import generate_replicate
from src.methods import PDWClosedForm
from src.simulation import CONFIG, build_conditions, make_check, replicate_rng

TIE_TOL = 1e-6   # within this of a threshold, the verdict could flip under rounding / solver error


def check_replicate(cond, rep, config, check, closed_form):
    """Run one replicate with every warning recorded; return one row of diagnostics."""
    rng = replicate_rng(config["master_seed"], cond["design"], cond["regime"],
                        cond["p"], cond["theta"], rep)
    row = {key: cond[key] for key in ("design", "regime", "p", "k", "theta", "n")}
    row.update({"rep": rep, "n_le_k": cond["n"] <= cond["k"], "exception": ""})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")              # record every warning, even repeated ones
        data = generate_replicate(cond["n"], cond["p"], cond["k"], config["sigma"],
                                  config["beta_min"], rng, design=cond["design"],
                                  toeplitz_mu=config["toeplitz_mu"].get(cond["design"]),
                                  sparse_q=config["sparse_q"])
        result = check.check(data, cond["lam"])
        try:
            closed = closed_form.check(data, cond["lam"])
            row["agree_closed_form"] = result["success"] == closed["success"]
        except np.linalg.LinAlgError as err:         # e.g. X_S^T X_S exactly singular
            row["agree_closed_form"] = np.nan
            row["exception"] = f"LinAlgError: {err}"

    # Condition number of X_S (2-norm); the closed form solves with X_S^T X_S, whose kappa is this squared
    row["cond_XS"] = np.nan if row["n_le_k"] else np.linalg.cond(data["X"][:, data["support"]])
    row["converged"] = result["solver_converged"]
    row["near_tie"] = bool(abs(result["max_abs_z"] - 1.0) < TIE_TOL
                           or 0.0 < abs(result["sign_margin"]) < TIE_TOL)
    row["n_warnings"] = len(caught)
    row["warnings"] = "; ".join(sorted({f"{w.category.__name__}: {w.message}" for w in caught}))
    return row


def summarize(df):
    """Counts per (design, regime) and the worst-conditioned conditions."""
    by_group = df.groupby(["design", "regime"]).agg(
        reps=("rep", "size"),
        n_le_k=("n_le_k", "sum"),
        not_converged=("converged", lambda s: int((s == False).sum())),   # NaN (n <= k) not counted
        with_warnings=("n_warnings", lambda s: int((s > 0).sum())),
        exceptions=("exception", lambda s: int((s != "").sum())),
        near_ties=("near_tie", "sum"),
        disagree=("agree_closed_form", lambda s: int((s == False).sum())),
        median_cond=("cond_XS", "median"),
        max_cond=("cond_XS", "max"),
    )
    worst = (df.groupby(["design", "regime", "p", "theta", "n", "k"])["cond_XS"].max()
               .sort_values(ascending=False).head(10))
    return by_group, worst


def parse_args():
    parser = argparse.ArgumentParser(description="Check numerical stability across all conditions.")
    parser.add_argument("--n-reps", type=int, default=10, help="replicates per condition")
    parser.add_argument("--out", default="results/profiling/stability_baseline.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    config = copy.deepcopy(CONFIG)
    check, closed_form = make_check(config), PDWClosedForm()

    rows = [check_replicate(cond, rep, config, check, closed_form)
            for cond in tqdm(build_conditions(config), desc="conditions", unit="cond")
            for rep in range(args.n_reps)]
    df = pd.DataFrame(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    by_group, worst = summarize(df)
    pd.set_option("display.width", 200)
    print("\nPer design and regime:\n", by_group.round(1).to_string())
    print("\nTen worst-conditioned conditions (max cond(X_S) over reps):\n", worst.to_string())
    print("\nDistinct warning messages:", df.loc[df["warnings"] != "", "warnings"].unique())


if __name__ == "__main__":
    main()
