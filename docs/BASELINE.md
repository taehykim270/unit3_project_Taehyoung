# Baseline Performance (Unit 2 code)

## 1. Setup and total runtime

**Hardware:** Apple M5, 10 cores, 16 GB RAM.  
**Workload:** full Unit 2 study: 4 designs x 3 sparsity regimes x 3 values of p x 24 values of theta.
= 864 conditions x 200 replicates = 172,800 replicates.  
**Command:** `/usr/bin/time -l make simulate` (single process)

| Measure | Value |
|---|---|
| Simulation loop (`elapsed_seconds`) | 446.4 s |
| Wall-clock, whole command (`real`) | 448.1 s |
| CPU time (`user` / `sys`) | 480.8 s / 1.8 s |
| Peak memory (max resident set size) | 1.18 GB |
| Solver convergence failures | 0 / 172,800 |

`user` > `real` shows that numpy's BLAS backend sometimes uses more than one core,
even though the simulation loop itself is serial.

## 2. Profiling
### 2.1 Function-level profile (cProfile)

Command: `python -m src.profile_sim` (all 864 conditions, 10 reps each).  
Full output: `results/profiling/baseline_cprofile.txt`.

| Component | cumtime (s) | Share |
|---|---:|---:|
| **Data generation** (`generate_replicate`) | 12.2 | 50% |
| - Toeplitz designs (`Z @ L.T`) | 6.8 | 27% |
| - sparse design | 2.8 | 12% |
| - Gaussian design | 2.2 | 9% |
| **PDW check** (`PrimalDualWitness.check`) | 12.2 | 50% |
| - sklearn `Lasso.fit` on the support | 7.7 | 31% |
|   - coordinate descent (`enet_path`) | 4.3 | 18% |
|   - input validation / overhead | ~3.3 | 14% |
| - numpy work inside `check` itself (tottime) | 4.1 | 17% |
| Seeding (`replicate_rng`) | 0.1 | <1% |

Observations:
- Generating the data costs as much as running the method.
- About 40% of the Lasso time is scikit-learn input validation, not optimization.

### 2.2 Line-level profile (line_profiler)

`python -m src.profile_sim --lines`.  
Full output: `results/profiling/baseline_lines.txt`.

| Function | Line | Share of function |
|---|---|---:|
| `check` | `restricted_solution(...)` (sklearn Lasso) | 67% |
| | `X[:, off_support].T @ residual` | 19% |
| | `X_S = X[:, support]` | 9% |
| `generate_toeplitz_design` | `rng.standard_normal((n, p))` | 64% |
| | `Z @ L.T` | 36% |
| `generate_sparse_design` | `rng.random((n, p))` | 68% |

**Main bottlenecks:** (1) the sklearn Lasso fit, (2) random number generation (~70% of data
generation time), (3) copying columns of `X` by fancy indexing in `check`, (4) the O(np^2)
Toeplitz product `Z @ L.T`.

## 3. Computational complexity
`python -m src.complexity` times one replicate (median of 5) in the linear regime, with data generation (DGP) and the PDW check timed separately.  
Raw times: `results/profiling/complexity_baseline.csv`.

**Theory** (T = number of coordinate-descent iterations):

| Step | Cost |
|---|---|
| draw X | O(np) |
| Toeplitz `Z @ L.T` | O(np^2) |
| sklearn Lasso on X_S | O(T nk) |
| `X[:, off_support].T @ residual` | O(n(p-k)) |

**Empirical log-log slopes:**

| Grid | Design | DGP | check | total |
|---|---|---:|---:|---:|
| n (p = 512) | identity | 0.99 | 0.49 | 0.67 |
| n (p = 512) | toeplitz | 0.83 | 0.35 | 0.58 |
| p (theta = 1) | identity | 2.15 | 1.65 | 1.82 |
| p (theta = 1) | toeplitz | 2.31 | 1.57 | 1.90 |

- n grid: data generation is linear in n as predicted. The check grows more slowly because the number of Lasso iterations falls as n grows (T = 82 at n = 235 vs 13 at n = 5636): near n = k the restricted problem is ill-conditioned.
- p grid: n and k grow with p, so the predicted slopes are about 2+ (identity, O(p^2 log p)) and 3+ (Toeplitz, O(p^3 log p)). The Toeplitz slope is only 2.31 because at these sizes random number generation still accounts for a large share of its time; the O(np^2) product's share grows with p (Toeplitz/identity DGP time ratio rises from 1.14 at p = 64 to 1.72 at p = 1024).


## 4. Numerical stability
(Step 1.5)
