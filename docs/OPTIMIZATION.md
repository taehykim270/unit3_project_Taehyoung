# Optimizations

Timings: `python -m src.simulation --n-reps 10` (all 864 conditions, 8,640 replicates), Apple M5.

| Step | Change | Time (s) | Speedup vs baseline |
|---|---|---:|---:|
| 0 | Baseline (Unit 2) | 22.3 | 1.00x |
| 1 | Closed-form PDW with Cholesky (replaces sklearn Lasso) | 17.0 | 1.31x |

## 1. Closed-form PDW with Cholesky and a singularity guard
- Verdicts identical to baseline on 8,640 / 8,640 replicates; `max_abs_z` differs by at most 4.8e-11 (sklearn's tolerance).
