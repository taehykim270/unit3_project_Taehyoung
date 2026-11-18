# Optimizations

Timings: `python -m src.simulation --n-reps 10` (all 864 conditions, 8,640 replicates), Apple M5.

| Step | Change | Time (s) | Speedup vs baseline |
|---|---|---:|---:|
| 0 | Baseline (Unit 2) | 22.3 | 1.00x |
| 1 | Closed-form PDW with Cholesky (replaces sklearn Lasso) | 17.0 | 1.31x |
| 2 | No column copies in `check` (`X.T @ r` + boolean mask) | 14.3 | 1.56x |
| 3 | + parallel over conditions (joblib, n_jobs = -1) | 6.8 | 3.28x |

**Correctness:** `tests/test_regression.py` runs the same small study with both versions. Since the
random draws are unchanged, it checks exact equality (per-replicate success/failure result, per-condition success
rates, serial = parallel) instead of a statistical test, plus the edge cases n <= k and singular X_S.

## 1. Closed-form PDW with Cholesky and a singularity guard
- Same success/failure result as the baseline on 8,640 / 8,640 replicates; `max_abs_z` differs by at most 4.8e-11 (sklearn's tolerance).

## 2. Avoid copying columns of X in the dual-feasibility step
- `X[:, off_support]` (fancy indexing) copies n x (p - k) numbers every replicate; `X.T` is a view.
- `timeit` at n = 5636, p = 512, k = 205: `X[:, off].T @ r` 3.52 ms vs `(X.T @ r)[mask]` 0.268 ms (13x).
- `np.setdiff1d` (sorts) replaced by a boolean mask.
- Same success/failure result as the baseline on 8,640 / 8,640 replicates.

## 2. Tried and rejected: Toeplitz design via AR(1) recursion
Sigma_ij = mu^|i-j| is an AR(1) covariance, so X can be built column by column with `scipy.signal.lfilter` in O(np) instead of the O(np^2) product `Z @ L.T`, using the same draws Z (X agrees to 1e-15). It is slower at the study's sizes (n = 5636, mu = 0.5):

| p | `Z @ L.T` | AR(1) | random draws alone |
|---:|---:|---:|---:|
| 256 | 6.1 ms | 9.1 ms | 4.5 ms |
| 512 | 14.7 ms | 17.3 ms | 9.1 ms |
| 1024 | 39.0 ms | 33.4 ms | 18.3 ms |
| 2048 | 125.2 ms | 66.8 ms | 36.5 ms |

The product runs in Apple's Accelerate BLAS, while the recursion is sequential, so AR(1) only wins above p of about 700. The study uses p <= 512, so the optimized code keeps `Z @ L.T` (`toeplitz_method="ar1"` remains available).

## 3. Parallelization across conditions (joblib)
- Conditions are independent and every replicate has its own seed, so results do not depend on the worker;
  joblib returns results in submission order, so the output is identical to the serial run
  (`assert_frame_equal` on all 8,640 rows).

| n_jobs | 1 | 2 | 4 | 6 | 8 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| Time (s), 10 reps | 14.4 | 10.3 | 6.8 | 6.3 | 6.6 | 7.1 |
| Speedup, 10 reps | 1.00 | 1.40 | 2.12 | 2.29 | 2.18 | 2.03 |
| Time (s), 50 reps | 72.4 | | 31.1 | | | 24.3 |
| Speedup, 50 reps | 1.00 | | 2.33 | | | 2.98 |

- The M5 has 4 performance and 6 efficiency cores. With 10 reps, each worker's start-up cost (new process, imports) dominates and more than 4 workers do not help; with 50 reps the efficiency cores pay off (10 workers: 2.98x vs 4 workers: 2.33x). `make parallel` uses all cores.
- Even 4 workers reach only 2.33x: all workers generate large random matrices at the same time and share the memory bandwidth.

## Differences from the Unit 2 results
Full study rerun with the optimized code (`make parallel`, 172,800 replicates) and compared with the Unit 2 tables:
- **Success rates are identical in all 864 conditions**; thresholds, curve collapse and the paper comparison agree up to rounding (<= 2e-16). Figures 1-5 (success curves) are unchanged.
- **PDW diagnostics differ on failed replicates** (`dual_feasible_rate` by up to 0.04, `max_abs_z` quantiles, numbers of violations and sign errors). When recovery fails, the baseline's restricted Lasso and the closed form (which assumes the true signs) produce different beta_check_S, so they certify the failure at different PDW steps. Both verdicts are correct by Lemma 3, and on successful replicates the diagnostics agree (max_abs_z within 1e-8, `tests/test_regression.py`). This changes `fig1_pdw_diagnostics` only. Singular X_S (n <= k, or rank-deficient X_S in the sparse design) now gives NaN diagnostics.