"""
Profile a reduced version of the simulation study with cProfile.

Run from the project root:
    python -m src.profile_sim                       # all conditions, 10 reps each
    python -m src.profile_sim --n-reps 5 --top 30
"""

import argparse
import copy
import cProfile
import pstats
from pathlib import Path

from src.simulation import CONFIG, run_simulation


def parse_args():
    parser = argparse.ArgumentParser(description="Profile the simulation")
    parser.add_argument("--n-reps", type=int, default=10, help="replicates per condition")
    parser.add_argument("--top", type=int, default=20, help="rows to print in each table")
    parser.add_argument("--out", default="results/profiling/baseline.prof",
                        help="where to save the raw profile")
    return parser.parse_args()


def main():
    args = parse_args()
    config = copy.deepcopy(CONFIG)
    config["n_reps"] = args.n_reps

    with cProfile.Profile() as profiler:
        run_simulation(config)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(args.out)

    stats = pstats.Stats(profiler).strip_dirs()
    print("=" * 30, "sorted by cumulative time", "=" * 30)
    stats.sort_stats("cumulative").print_stats(args.top)
    print("=" * 30, "sorted by own time (tottime)", "=" * 30)
    stats.sort_stats("tottime").print_stats(args.top)


if __name__ == "__main__":
    main()
