"""
Profile a reduced version of the simulation study with cProfile.

Run from the project root:
    python -m src.profile_sim                       # all conditions, 10 reps each
    python -m src.profile_sim --n-reps 5 --top 30
    python -m src.profile_sim --lines               # line-by-line profile of the slowest functions
"""

import argparse
import copy
import cProfile
import pstats
from pathlib import Path

from src.simulation import CONFIG, IMPLS, run_simulation


def parse_args():
    parser = argparse.ArgumentParser(description="Profile the simulation")
    parser.add_argument("--n-reps", type=int, default=10, help="replicates per condition")
    parser.add_argument("--top", type=int, default=20, help="rows to print in each table")
    parser.add_argument("--impl", choices=sorted(IMPLS), default="baseline", help="code version to profile")
    parser.add_argument("--out", default=None, help="raw profile file (default: results/profiling/<impl>.prof)")
    parser.add_argument("--lines", action="store_true",
                        help="line-by-line profile of the hot functions instead of cProfile")   # helps are for my own understanding
    return parser.parse_args()


def line_profile(config):
    """Time every line of the functions that cProfile flagged as expensive."""
    from line_profiler import LineProfiler
    from src import dgps, methods

    profiler = LineProfiler()
    profiler.add_function(methods.PrimalDualWitness.check)
    profiler.add_function(dgps.generate_replicate)
    profiler.add_function(dgps.generate_toeplitz_design)
    profiler.add_function(dgps.generate_sparse_design)
    profiler.add_function(methods.PDWFast.check)

    profiler.runcall(run_simulation, config)
    profiler.print_stats(output_unit=1e-3)       # Report in milliseconds


def main():
    args = parse_args()
    config = copy.deepcopy(CONFIG)
    config["n_reps"] = args.n_reps
    config.update(IMPLS[args.impl])
    out = args.out or f"results/profiling/{args.impl}.prof"

    if args.lines:
        line_profile(config)
        return

    with cProfile.Profile() as profiler:
        run_simulation(config)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(out)

    stats = pstats.Stats(profiler).strip_dirs()
    print("=" * 30, "sorted by cumulative time", "=" * 30)
    stats.sort_stats("cumulative").print_stats(args.top)
    print("=" * 30, "sorted by own time (tottime)", "=" * 30)
    stats.sort_stats("tottime").print_stats(args.top)


if __name__ == "__main__":
    main()
