"""Regression tests (Unit 3): the optimized code gives the same answers as the Unit 2 baseline."""

import numpy as np
import pandas as pd
import pytest

from helpers import small_config
from src.dgps import generate_replicate, generate_toeplitz_design, generate_toeplitz_design_ar1
from src.methods import PDWFast, PDWRestrictedLasso, regularization_lambda
from src.simulation import IMPLS, run_simulation

# theta = 0.1 gives n <= k (linear regime) and theta = 0.2 gives singular X_S (sparse design)
GRID = dict(n_reps=10, thetas=[0.1, 0.2, 0.5, 1.0, 2.0])


@pytest.fixture(scope="module")
def baseline_and_optimized():
    """The same small study (all designs and regimes, p = 128), run with both versions of the code."""
    base = run_simulation(small_config(**GRID, **IMPLS["baseline"]))
    opt = run_simulation(small_config(**GRID, **IMPLS["optimized"]))
    return base, opt


def test_same_success_result_for_every_replicate(baseline_and_optimized):
    """Same seed -> same data -> same success/failure, replicate by replicate."""
    base, opt = baseline_and_optimized
    assert (base["seed_key"] == opt["seed_key"]).all()
    assert (base["success"] == opt["success"]).all()


def test_success_rates_match(baseline_and_optimized):
    """Per-condition success rates (what the figures show) are identical."""
    base, opt = baseline_and_optimized
    keys = ["design", "regime", "p", "theta"]
    pd.testing.assert_series_equal(base.groupby(keys)["success"].mean(),
                                   opt.groupby(keys)["success"].mean())


def test_max_abs_z_matches_where_both_succeed(baseline_and_optimized):
    """Where the PDW succeeds, max |z_j| agrees up to the baseline solver's tolerance (1e-10)."""
    base, opt = baseline_and_optimized
    ok = base["success"] & opt["success"]
    assert ok.sum() > 50
    np.testing.assert_allclose(opt.loc[ok, "max_abs_z"], base.loc[ok, "max_abs_z"], atol=1e-8)


def test_parallel_identical_to_serial():
    """Parallel run (2 workers) gives exactly the same table as the serial run."""
    config = small_config(**GRID, **IMPLS["optimized"])
    pd.testing.assert_frame_equal(run_simulation(config), run_simulation(dict(config, n_jobs=2)))


@pytest.mark.parametrize("mu", [0.1, 0.5])
def test_ar1_toeplitz_equals_cholesky(mu):
    """The AR(1) recursion gives the same X as Z @ L.T from the same draws (rejected optimization)."""
    a = generate_toeplitz_design(200, 64, mu, np.random.default_rng(3))
    b = generate_toeplitz_design_ar1(200, 64, mu, np.random.default_rng(3))
    np.testing.assert_allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("kind", ["zero_column", "duplicate_column"])
def test_singular_X_S_fails_like_baseline(kind):
    """Rank-deficient X_S: both versions report failure, and the optimized one raises no error."""
    n, p, k = 60, 128, 16
    d = generate_replicate(n, p, k, 0.5, 0.5, np.random.default_rng(5))
    S = d["support"]
    if kind == "zero_column":                  # what happens in the sparse design at small n
        d["X"][:, S[0]] = 0.0
    else:
        d["X"][:, S[1]] = d["X"][:, S[0]]
    d["y"] = d["X"] @ d["beta"] + d["w"]
    lam = regularization_lambda(n, p, k, 0.5)
    assert PDWRestrictedLasso().check(d, lam)["success"] is False
    assert PDWFast().check(d, lam)["success"] is False


def test_n_le_k_same_output():
    """n <= k: both versions return failure with undefined (NaN) diagnostics."""
    n, p, k = 16, 128, 16
    d = generate_replicate(n, p, k, 0.5, 0.5, np.random.default_rng(6))
    lam = regularization_lambda(n, p, k, 0.5)
    base, fast = PDWRestrictedLasso().check(d, lam), PDWFast().check(d, lam)
    assert base.keys() == fast.keys()
    assert base["success"] is False and fast["success"] is False
    assert np.isnan(base["max_abs_z"]) and np.isnan(fast["max_abs_z"])
