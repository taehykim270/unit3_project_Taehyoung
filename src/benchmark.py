"""
Benchmark: total time of the simulation study, baseline vs optimized, and speedup vs number of workers.

Run from the project root:
    python -m src.benchmark                      # 20 reps per condition
    python -m src.benchmark --n-reps 10 --jobs 1 4 10
"""

import argparse
import copy
import time
from pathlib import Path

import pandas as pd

from src.simulation import CONFIG, IMPLS, run_simulation


def time_run(impl, n_jobs, n_reps):
    """Wall-clock seconds to run all 864 conditions with n_reps replicates each."""
    config = copy.deepcopy(CONFIG)
    config.update(IMPLS[impl], n_jobs=n_jobs, n_reps=n_reps)
    start = time.perf_counter()
    run_simulation(config)
    return time.perf_counter() - start


def parse_args():
    parser = argparse.ArgumentParser(description="Time baseline vs optimized code.")
    parser.add_argument("--n-reps", type=int, default=20, help="replicates per condition")
    parser.add_argument("--jobs", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10],
                        help="numbers of workers to try for the optimized code")
    parser.add_argument("--out", default=None, help="default: results/profiling/benchmark_<n_reps>reps.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    jobs = sorted(set([1] + args.jobs))            # 1 worker is always needed as the reference
    runs = [("baseline", 1)] + [("optimized", j) for j in jobs]

    rows = []
    for impl, n_jobs in runs:
        seconds = time_run(impl, n_jobs, args.n_reps)
        rows.append({"impl": impl, "n_jobs": n_jobs, "n_reps": args.n_reps, "seconds": seconds})
        print(f"{impl:9s} n_jobs={n_jobs:2d}: {seconds:7.1f} s")

    df = pd.DataFrame(rows)
    baseline = df.loc[df["impl"] == "baseline", "seconds"].iloc[0]
    serial = df.loc[(df["impl"] == "optimized") & (df["n_jobs"] == 1), "seconds"].iloc[0]
    df["speedup_vs_baseline"] = baseline / df["seconds"]
    df["speedup_vs_serial"] = serial / df["seconds"]       # parallel speedup of the optimized code

    out = Path(args.out or f"results/profiling/benchmark_{args.n_reps}reps.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\n" + df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
