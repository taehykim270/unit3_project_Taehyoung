"""
Empirical computational complexity of one simulation replicate.

Times DGP and the PDW check separately, on two grids:
    - n grid: p = 512 fixed, n varies through theta
    - p grid: theta = 1 fixed, p varies (k and n grow with p)
then fits log(time) = a + b * log(x); the slope b estimates the exponent in time.

Run from the project root:
    python -m src.complexity
"""

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.dgps import generate_replicate, sample_size, sparsity_k
from src.methods import regularization_lambda
from src.simulation import CONFIG, make_check

DESIGNS = ["identity", "toeplitz"]
REGIME = "linear"
N_GRID = {"p": 512, "thetas": [0.1, 0.2, 0.4, 0.8, 1.6, 2.4]}
P_GRID = {"theta": 1.0, "p_values": [64, 128, 256, 512, 1024]}


def time_replicate(n, p, k, design, config, check, rng, n_repeat):
    """Median wall-clock seconds of data generation and of the PDW check over n_repeat replicates."""
    lamda = regularization_lambda(n, p, k, config["sigma"])
    t_dgp, t_check = [], []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        data = generate_replicate(n, p, k, config["sigma"], config["beta_min"], rng, design=design,
                                  toeplitz_mu=config["toeplitz_mu"].get(design),
                                  sparse_q=config["sparse_q"])
        t1 = time.perf_counter()
        check.check(data, lamda)
        t2 = time.perf_counter()
        t_dgp.append(t1 - t0)
        t_check.append(t2 - t1)
    return float(np.median(t_dgp)), float(np.median(t_check))


def run_grids(config, n_repeat):
    """Time every (design, grid point); return one row per point."""
    check = make_check(config)
    rng = np.random.default_rng(config["master_seed"])
    points = [("n", N_GRID["p"], theta) for theta in N_GRID["thetas"]]
    points += [("p", p, P_GRID["theta"]) for p in P_GRID["p_values"]]

    rows = []
    for design in DESIGNS:
        for grid, p, theta in points:
            k = sparsity_k(p, REGIME, config["alpha"], config["delta"])
            n = sample_size(theta, p, k)
            t_dgp, t_check = time_replicate(n, p, k, design, config, check, rng, n_repeat)
            rows.append({"grid": grid, "design": design, "p": p, "k": k, "theta": theta, "n": n,
                         "t_dgp": t_dgp, "t_check": t_check, "t_total": t_dgp + t_check})
            print(f"{design:9s} grid={grid}  p={p:5d}  k={k:4d}  n={n:6d}  "
                  f"dgp={1e3 * t_dgp:8.2f} ms  check={1e3 * t_check:8.2f} ms")
    return pd.DataFrame(rows)


def fit_slopes(df):
    """Slope of log(time) on log(n) (n grid) or log(p) (p grid), per design and component."""
    rows = []
    for (grid, design), g in df.groupby(["grid", "design"]):
        x = np.log(g["n"] if grid == "n" else g["p"])
        row = {"grid": grid, "design": design}
        for col in ("t_dgp", "t_check", "t_total"):
            row[col] = np.polyfit(x, np.log(g[col]), 1)[0]    # polyfit returns [slope, intercept]
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Time one replicate as a function of n and p.")
    parser.add_argument("--n-repeat", type=int, default=5, help="replicates timed per point")
    parser.add_argument("--out", default="results/profiling/complexity_baseline.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    config = copy.deepcopy(CONFIG)

    df = run_grids(config, args.n_repeat)
    slopes = fit_slopes(df)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    slopes.to_csv(out.with_name(out.stem + "_slopes.csv"), index=False)

    print("\nLog-log slopes (time ~ x^slope):")
    print(slopes.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
    