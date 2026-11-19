# Optimizations

Starting point: the Unit 2 code (tag `unit2-baseline`); measurements in [BASELINE.md](BASELINE.md).
Hardware: Apple M5 (4 performance + 6 efficiency cores), 16 GB RAM.

## Summary

| Version | Full study (172,800 replicates) | Speedup |
|---|---:|---:|
| Baseline (Unit 2), 1 core | 446.4 s | 1.00x |
| Optimized, all cores (`make parallel`) | **100.4 s** | **4.45x** |

Step by step (`python -m src.simulation --n-reps 10`, all 864 conditions, 8,640 replicates):

| Step | Change | Category | Time (s) | Speedup |
|---|---|---|---:|---:|
| 0 | Baseline (Unit 2) | | 22.3 | 1.00x |
| 1 | Closed-form PDW with Cholesky and a singularity guard | algorithmic, numerical stability | 17.0 | 1.31x |
| 2 | No column copies in the dual step | algorithmic / array programming | 14.3 | 1.56x |
| 2b | Baseline solver as fallback when the closed-form signs are wrong (all outputs = baseline) | numerical stability | 16.3 | 1.37x |
| 3 | Parallel conditions (joblib, all cores) | parallelization | 6.8 | 3.28x |
| - | Toeplitz design via AR(1) recursion | tried, rejected | slower | - |

![Total time and parallel speedup](../results/figures/perf_speedup.png)

**Correctness:** `tests/test_regression.py` checks that both versions give the same results on every
replicate (success/failure and all diagnostics, up to the solver tolerance), that serial = parallel,
and the edge cases n <= k and singular X_S. Differences in the full study are explained [at the end](#differences-from-the-unit-2-results).

---

## 1. Closed-form PDW with Cholesky and a singularity guard

**Problem.** The sklearn Lasso fit on $X_S$ took 31% of the time (cProfile). A closed-form solve with
`np.linalg.solve` raises `LinAlgError` when $X_S$ is singular (sparse design, small $n$).

**Solution.** With the signs fixed to $\mathrm{sign}(\beta^*_S)$, Step 1 of the PDW is the linear system

$$
(X_S^\top X_S)\,\check{\beta}_S = X_S^\top y - n\lambda_n\,\mathrm{sign}(\beta^*_S),
$$

which gives the same success/failure result as the restricted Lasso (Lemma 3). It is solved by a
Cholesky factorization $X_S^\top X_S = LL^\top$. If the solution has exactly the signs
$\mathrm{sign}(\beta^*_S)$, it is the restricted Lasso solution (it meets the Lasso's optimality
condition, which has a unique solution when $X_S$ has full rank). Otherwise (a sign is wrong, or
$X_S^\top X_S$ is singular) the baseline solver is run instead, so every output equals the baseline's. New class `PDWFast` in `src/methods.py`.

```python
# before (PDWRestrictedLasso): iterative solver
beta_check_S, converged = _lasso_fit_status(X_S, y, lam, tol=1e-10)   # sklearn Lasso(...).fit(X_S, y)

# after (PDWFast): one linear solve, baseline solver only as fallback
try:
    chol = cho_factor(X_S.T @ X_S)
except np.linalg.LinAlgError:              # X_S rank-deficient
    return self.fallback.check(data, lam)
beta_check_S = cho_solve(chol, X_S.T @ y - n * lam * z_S)
if np.min(z_S * beta_check_S) <= self.fallback.sign_tol:   # a sign is wrong
    return self.fallback.check(data, lam)
```

**Impact.** 22.3 s -> 17.0 s (1.31x). Check time at n = 5636, p = 512: 19.3 ms -> 3.7 ms per
replicate. No exceptions on singular $X_S$. About 25% of replicates use the fallback (mostly
$\theta \le 1$, where $n$ is small), which costs 14% (14.3 s -> 16.3 s, step 2b) but makes every output
equal to the baseline's.

**Trade-offs.** Forming $X_S^\top X_S$ squares the condition number, $\kappa(X_S^\top X_S) = \kappa(X_S)^2$;
a QR factorization of $X_S$ would avoid this at about twice the cost. Without the fallback the code
would be 14% faster, but the PDW diagnostics of failed replicates would differ from the baseline.

## 2. No column copies in the dual-feasibility step

**Problem.** line_profiler: `X[:, off_support].T @ residual` took 19% of `check`. Fancy indexing
`X[:, off_support]` copies n x (p - k) numbers every replicate.

**Solution.** Multiply with the full `X.T` (a view, no copy) and drop the support entries afterwards;
replace `setdiff1d` by a boolean mask.

```python
# before
off_support = np.setdiff1d(np.arange(p), support)
z_Sc = X[:, off_support].T @ residual / (n * lam)

# after
off_support = np.ones(p, dtype=bool)
off_support[support] = False
z_Sc = (X.T @ residual)[off_support] / (n * lam)
```

**Impact.** `timeit` at n = 5636, p = 512, k = 205: 3.52 ms -> 0.268 ms (13x) for this line.
Whole run 17.0 s -> 14.3 s. Same success/failure result on 8,640 / 8,640 replicates.

**Trade-offs.** Computes k extra entries that are thrown away (k/p of the product); negligible.

## 3. Parallelization across conditions (joblib)

**Problem.** After steps 1-2b, about 70% of the time is data generation, mostly random number generation.
It cannot be made cheaper without changing the random draws, but the 864 conditions are independent.

**Solution.** Run the conditions in worker processes with `joblib.Parallel` (`--n-jobs`). Every replicate
has its own seed, and joblib returns results in submission order, so the output table is identical to
the serial run.

```python
# before
for cond in tqdm(conditions):
    records.extend(run_condition(cond, config))

# after
results = Parallel(n_jobs=config["n_jobs"], return_as="generator")(
    delayed(run_condition)(cond, config) for cond in conditions)
```

**Impact** (`make benchmark`, 20 reps per condition):

| n_jobs | 1 | 2 | 4 | 6 | 8 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| Time (s) | 32.9 | 22.5 | 14.5 | 12.3 | 11.4 | 10.6 |
| Speedup vs 1 worker | 1.00 | 1.46 | 2.27 | 2.67 | 2.89 | 3.10 |

The speedup is far below the number of workers: the M5 has only 4 performance cores, each worker
pays a start-up cost, and all workers share the memory bandwidth.

**Trade-offs.** One more dependency (joblib); errors inside workers are harder to debug; memory use
grows with the number of workers.

## Tried and rejected: Toeplitz design via AR(1) recursion

Sigma_ij = mu^|i-j| is an AR(1) covariance, so X can be built column by column with
`scipy.signal.lfilter` in O(np) instead of the O(np^2) product `Z @ L.T`, from the same draws Z.
It is slower at the study's sizes (n = 5636, mu = 0.5):

| p | `Z @ L.T` | AR(1) |
|---:|---:|---:|
| 256 | 6.1 ms | 9.1 ms |
| 512 | 14.7 ms | 17.3 ms |
| 1024 | 39.0 ms | 33.4 ms |
| 2048 | 125.2 ms | 66.8 ms |

The product runs in Apple's Accelerate BLAS, while the recursion is sequential, so AR(1) only wins
above p of about 700. The study uses p <= 512, so the optimized code keeps `Z @ L.T`
(`toeplitz_method="ar1"` remains available).

---

## Profiling evidence

cProfile, all 864 conditions with 10 reps each (`make profile`):

| Component | Baseline (s) | Optimized (s) |
|---|---:|---:|
| Data generation | 12.2 | 12.4 |
| PDW check | 12.2 | **4.7** (2.2 of it in the fallback) |
| Total | 24.6 | 17.4 |
| Function calls | 14.2 million | 4.7 million |

Time of one replicate vs n and vs p (`make complexity`). For theta >= 0.4 the optimized check is
5-8x faster (n = 5636: 19.3 ms -> 4.0 ms) and grows linearly in n. At theta <= 0.2 most replicates
use the fallback, so the check costs about the same as the baseline there. Data generation is
unchanged and is now the largest cost.

![Complexity, baseline vs optimized](../results/figures/perf_complexity.png)

## Lessons learned

- **Best return on investment:** the closed form removed the largest single cost, and the copy fix
  (2 lines) made its line 13x faster. Parallelization gave the largest factor (3.10x) with little code,
  because the Unit 2 seeding rule (one seed per replicate) already made the results independent of
  execution order.
- **Surprises:** half of the baseline time was spent generating data, mostly drawing random numbers, not
  fitting the Lasso; about 40% of the Lasso time was scikit-learn checking its inputs.
- **Not worth the effort:** the AR(1) Toeplitz generator has the better complexity but is slower at
  p <= 512.

## Differences from the Unit 2 results

Full study rerun with the optimized code (`make parallel`, 172,800 replicates) and compared with the Unit 2 tables:
- **All tables agree with Unit 2**, including the PDW diagnostics: every column of `summary.csv`,
  `thresholds.csv`, `collapse.csv` and `paper_comparison.csv` differs by at most 7.9e-11 (the
  baseline solver's tolerance). All figures are unchanged.
- An earlier version without the fallback (always the closed form) gave identical success rates but
  different diagnostics on failed replicates, because the closed form assumes the true signs; the
  fallback (step 2b) removed this difference.
