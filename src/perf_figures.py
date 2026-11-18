"""
Performance figures (Unit 3), from the CSVs written by `make complexity` and `make benchmark`.

    perf_complexity   time of one replicate vs n and vs p (log-log), baseline vs optimized,
                      data generation and check shown separately
    perf_speedup      total time baseline vs optimized, and parallel speedup vs number of workers

Run from the project root:
    python -m src.perf_figures
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter

from src.figures import INK, MUTED, REFERENCE, save_figure, set_style

PROF = Path("results/profiling")
IMPL_COLORS = {"baseline": "#eb6834", "optimized": "#2a78d6"}
COMPONENTS = {"t_check": ("check", "-", "o"),               # column: (label, line style, marker)
              "t_dgp": ("data generation", "--", "s")}
X_TICKS = {"n": [250, 500, 1000, 2000, 5000], "p": [64, 128, 256, 512, 1024]}
LEGEND_LOC = {"n": "lower right", "p": "upper left"}         # corners left empty by the data
GRID_AXIS = {"n": ("n", "sample size $n$  ($p = 512$)"),
             "p": ("p", r"dimension $p$  ($\theta = 1$; $n$ and $k$ grow with $p$)")}
N_PERF_CORES = 4                                            # Apple M5: 4 performance + 6 efficiency cores

logger = logging.getLogger(__name__)


def load_complexity():
    """Both complexity CSVs in one table, with an `impl` column."""
    frames = [pd.read_csv(PROF / f"complexity_{impl}.csv").assign(impl=impl) for impl in IMPL_COLORS]
    return pd.concat(frames, ignore_index=True)


def fig_complexity(df, out_stem):
    """Rows: n grid, p grid. Columns: design. Markers: measured times; lines: log-log fits."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for row, grid in enumerate(GRID_AXIS):
        x_col, x_label = GRID_AXIS[grid]
        for col, design in enumerate(["identity", "toeplitz"]):
            ax = axes[row, col]
            for impl, color in IMPL_COLORS.items():
                g = df[(df["grid"] == grid) & (df["design"] == design) & (df["impl"] == impl)].sort_values(x_col)
                x = g[x_col].to_numpy(dtype=float)
                for t_col, (label, linestyle, marker) in COMPONENTS.items():
                    y = g[t_col].to_numpy() * 1e3                               # milliseconds
                    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
                    ax.plot(x, y, linestyle="none", marker=marker, markersize=5, color=color)
                    ax.plot(x, np.exp(intercept) * x**slope, linestyle=linestyle, color=color,
                            label=f"{impl}, {label} (slope {slope:.2f})")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())              # no crowded labels like 3x10^2
            ax.set_xticks(X_TICKS[grid], labels=[str(t) for t in X_TICKS[grid]])
            if grid == "n":
                ax.set_ylim(bottom=ax.get_ylim()[0] / 10)              # room for the legend below the data
            ax.set_xlabel(x_label)
            ax.set_ylabel("time per replicate (ms)")
            ax.set_title(f"{design} design, linear regime", color=INK)
            ax.legend(fontsize=8, loc=LEGEND_LOC[grid])
    fig.suptitle("Time of one replicate, baseline vs optimized (markers: median of 5; lines: log-log fit)",
                 fontsize=11, color=INK)
    fig.tight_layout()
    save_figure(fig, out_stem)


def fig_speedup(bench, out_stem):
    """Left: total time of the study. Right: speedup of the optimized code vs number of workers."""
    base = bench[bench["impl"] == "baseline"].iloc[0]
    opt = bench[bench["impl"] == "optimized"].sort_values("n_jobs")
    one = opt.iloc[0]                                        # n_jobs = 1
    best = opt.loc[opt["seconds"].idxmin()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    labels = ["baseline\n(Unit 2)", "optimized\n1 worker", f"optimized\n{int(best['n_jobs'])} workers"]
    times = [base["seconds"], one["seconds"], best["seconds"]]
    bars = ax1.bar(labels, times, color=[IMPL_COLORS["baseline"]] + 2 * [IMPL_COLORS["optimized"]])
    for bar, t in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width() / 2, t, f"{t:.1f} s\n({base['seconds'] / t:.2f}x)",
                 ha="center", va="bottom", fontsize=9, color=INK)
    ax1.set_ylim(0, 1.25 * max(times))
    ax1.set_ylabel("total time (s)")
    ax1.set_title(f"Whole study, {int(base['n_reps'])} reps per condition", color=INK)
    ax1.grid(axis="x", visible=False)

    ax2.plot(opt["n_jobs"], opt["speedup_vs_serial"], marker="o", color=IMPL_COLORS["optimized"], label="measured")
    ax2.plot(opt["n_jobs"], opt["n_jobs"], linestyle="--", color=REFERENCE, label="ideal (speedup = workers)")
    ax2.axvline(N_PERF_CORES, linestyle=":", color=MUTED, label=f"{N_PERF_CORES} performance cores")
    ax2.set_xlabel("number of workers (n_jobs)")
    ax2.set_ylabel("speedup vs 1 worker")
    ax2.set_title("Parallel speedup of the optimized code", color=INK)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    save_figure(fig, out_stem)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Make the performance figures.")
    parser.add_argument("--bench-reps", type=int, default=20, help="plot results/profiling/benchmark_<n>reps.csv")
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()

    set_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_complexity(load_complexity(), out_dir / "perf_complexity")
    fig_speedup(pd.read_csv(PROF / f"benchmark_{args.bench_reps}reps.csv"), out_dir / "perf_speedup")


if __name__ == "__main__":
    main()
