# Baseline Performance (Unit 2 code)

## 1. Setup and total runtime

**Hardware:** Apple M5, 10 cores, 16 GB RAM.
**Workload:** full Unit 2 study: 4 designs x 3 sparsity regimes x 3 values of p x 24 values of theta
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
(Step 1.2-1.3)

## 3. Computational complexity
(Step 1.4)

## 4. Numerical stability
(Step 1.5)
