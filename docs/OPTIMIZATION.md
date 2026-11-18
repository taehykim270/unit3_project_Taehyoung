# Optimizations

Timings: `python -m src.simulation --n-reps 10` (all 864 conditions, 8,640 replicates), Apple M5.

| Step | Change | Time (s) | Speedup vs baseline |
|---|---|---:|---:|
| 0 | Baseline (Unit 2) | 22.3 | 1.00x |
| 1 | Closed-form PDW with Cholesky (replaces sklearn Lasso) | 17.0 | 1.31x |
| 2 | No column copies in `check` (`X.T @ r` + boolean mask) | 14.3 | 1.56x |

## 1. Closed-form PDW with Cholesky and a singularity guard
- Verdicts identical to baseline on 8,640 / 8,640 replicates; `max_abs_z` differs by at most 4.8e-11 (sklearn's tolerance).

## 2. Avoid copying columns of X in the dual-feasibility step
- `X[:, off_support]` (fancy indexing) copies n x (p - k) numbers every replicate; `X.T` is a view.
- `timeit` at n = 5636, p = 512, k = 205: `X[:, off].T @ r` 3.52 ms vs `(X.T @ r)[mask]` 0.268 ms (13x).
- `np.setdiff1d` (sorts) replaced by a boolean mask.
- Verdicts identical to baseline on 8,640 / 8,640 replicates.

## Tried and rejected: Toeplitz design via AR(1) recursion
Sigma_ij = mu^|i-j| is an AR(1) covariance, so X can be built column by column with `scipy.signal.lfilter`
in O(np) instead of the O(np^2) product `Z @ L.T`, using the same draws Z (X agrees to 1e-15).
It is slower at the study's sizes (n = 5636, mu = 0.5):

| p | `Z @ L.T` | AR(1) | random draws alone |
|---:|---:|---:|---:|
| 256 | 6.1 ms | 9.1 ms | 4.5 ms |
| 512 | 14.7 ms | 17.3 ms | 9.1 ms |
| 1024 | 39.0 ms | 33.4 ms | 18.3 ms |
| 2048 | 125.2 ms | 66.8 ms | 36.5 ms |

The product runs in Apple's Accelerate BLAS, while the recursion is sequential, so AR(1) only
wins above p of about 700. The study uses p <= 512, so the optimized code keeps `Z @ L.T`
(`toeplitz_method="ar1"` remains available).
