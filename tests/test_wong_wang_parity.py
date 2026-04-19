# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Parity tests: Python primary vs Rust batch (Wong-Wang)

"""Bit-exact parity between `WongWangUnit.step` and `py_wong_wang_simulate`.

The Python primary draws two `np.random.randn()` samples per step (one
for each pool). The Rust batch takes an `xi` array of length `2 *
n_steps` pre-drawn by the caller. Both consume the same RNG sequence
when the caller re-seeds and draws `2*N` samples before calling Rust,
so trajectories must match to within f64 round-off.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.wong_wang import WongWangUnit

# Rust backend availability — skip entire module if engine not built.
pytest.importorskip(
    "sc_neurocore_engine", reason="Rust engine wheel not installed (maturin develop)"
)
from sc_neurocore_engine import py_wong_wang_simulate  # noqa: E402


DEFAULT_PARAMS = dict(
    tau_s=0.1,
    gamma=0.641,
    j_n=0.2609,
    j_cross=0.0497,
    i_0=0.3255,
    sigma=0.02,
    dt=0.001,
)


def _run_python(n_steps: int, stim1: np.ndarray, stim2: np.ndarray, seed: int):
    """Run the Python primary step-by-step, collecting per-step traces."""
    np.random.seed(seed)
    unit = WongWangUnit(**DEFAULT_PARAMS)
    s1 = np.empty(n_steps, dtype=np.float64)
    s2 = np.empty(n_steps, dtype=np.float64)
    r1 = np.empty(n_steps, dtype=np.float64)
    r2 = np.empty(n_steps, dtype=np.float64)
    for t in range(n_steps):
        r1t, r2t = unit.step(float(stim1[t]), float(stim2[t]))
        s1[t], s2[t], r1[t], r2[t] = unit.s1, unit.s2, r1t, r2t
    return s1, s2, r1, r2


def _run_rust(n_steps: int, stim1: np.ndarray, stim2: np.ndarray, seed: int):
    """Run the Rust batch with xi drawn from the SAME seed. 2 draws per step,
    same order as the Python loop (xi1 then xi2)."""
    np.random.seed(seed)
    xi = np.random.randn(2 * n_steps).astype(np.float64)
    out = py_wong_wang_simulate(
        0.1,
        0.1,
        DEFAULT_PARAMS["tau_s"],
        DEFAULT_PARAMS["gamma"],
        DEFAULT_PARAMS["j_n"],
        DEFAULT_PARAMS["j_cross"],
        DEFAULT_PARAMS["i_0"],
        DEFAULT_PARAMS["sigma"],
        DEFAULT_PARAMS["dt"],
        stim1.astype(np.float64),
        stim2.astype(np.float64),
        xi,
    )
    return out["s1"], out["s2"], out["r1"], out["r2"]


class TestBitExactParity:
    """Python step-by-step RNG order and Rust pre-drawn xi MUST match.

    Tolerance is set by f64 round-off over the non-linear `phi`
    transfer; empirical check says `1e-12` is comfortable for `N=10k`.
    """

    def test_parity_quiescent(self):
        n = 10_000
        stim1 = np.zeros(n)
        stim2 = np.zeros(n)
        p1, p2, pr1, pr2 = _run_python(n, stim1, stim2, seed=42)
        r1, r2, rr1, rr2 = _run_rust(n, stim1, stim2, seed=42)
        assert np.allclose(p1, r1, atol=1e-12, rtol=0)
        assert np.allclose(p2, r2, atol=1e-12, rtol=0)
        assert np.allclose(pr1, rr1, atol=1e-9, rtol=0)
        assert np.allclose(pr2, rr2, atol=1e-9, rtol=0)

    def test_parity_biased(self):
        n = 10_000
        stim1 = np.full(n, 0.1)
        stim2 = np.zeros(n)
        p1, p2, pr1, pr2 = _run_python(n, stim1, stim2, seed=123)
        r1, r2, rr1, rr2 = _run_rust(n, stim1, stim2, seed=123)
        assert np.allclose(p1, r1, atol=1e-12, rtol=0)
        assert np.allclose(p2, r2, atol=1e-12, rtol=0)

    def test_parity_symmetric(self):
        n = 10_000
        stim1 = np.full(n, 0.05)
        stim2 = np.full(n, 0.05)
        p1, p2, _, _ = _run_python(n, stim1, stim2, seed=7)
        r1, r2, _, _ = _run_rust(n, stim1, stim2, seed=7)
        assert np.allclose(p1, r1, atol=1e-12, rtol=0)
        assert np.allclose(p2, r2, atol=1e-12, rtol=0)

    def test_parity_across_seeds(self):
        """Parity holds for every seed, not just one."""
        n = 5_000
        stim1 = np.full(n, 0.1)
        stim2 = np.zeros(n)
        for seed in (0, 1, 42, 100, 2025):
            p1, p2, _, _ = _run_python(n, stim1, stim2, seed=seed)
            r1, r2, _, _ = _run_rust(n, stim1, stim2, seed=seed)
            assert np.allclose(p1, r1, atol=1e-12, rtol=0), f"seed={seed}: s1 diverged"
            assert np.allclose(p2, r2, atol=1e-12, rtol=0), f"seed={seed}: s2 diverged"


class TestFinalStateParity:
    """Rust-returned `s1_final` / `s2_final` must match the last trace sample
    and the Python primary's final state."""

    def test_rust_s1_final_matches_trace_last(self):
        n = 2_000
        stim1 = np.full(n, 0.1)
        stim2 = np.zeros(n)
        np.random.seed(0)
        xi = np.random.randn(2 * n).astype(np.float64)
        out = py_wong_wang_simulate(
            0.1,
            0.1,
            DEFAULT_PARAMS["tau_s"],
            DEFAULT_PARAMS["gamma"],
            DEFAULT_PARAMS["j_n"],
            DEFAULT_PARAMS["j_cross"],
            DEFAULT_PARAMS["i_0"],
            DEFAULT_PARAMS["sigma"],
            DEFAULT_PARAMS["dt"],
            stim1,
            stim2,
            xi,
        )
        assert out["s1_final"] == out["s1"][-1]
        assert out["s2_final"] == out["s2"][-1]

    def test_python_rust_final_state_match(self):
        n = 3_000
        stim1 = np.full(n, 0.15)
        stim2 = np.full(n, 0.05)
        p1, p2, _, _ = _run_python(n, stim1, stim2, seed=99)
        r1, r2, _, _ = _run_rust(n, stim1, stim2, seed=99)
        assert abs(p1[-1] - r1[-1]) < 1e-12
        assert abs(p2[-1] - r2[-1]) < 1e-12


class TestInputValidation:
    """Python wrapper must raise ValueError on shape mismatches, not panic."""

    def test_stim1_stim2_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="stim1 and stim2 length"):
            py_wong_wang_simulate(
                0.1,
                0.1,
                DEFAULT_PARAMS["tau_s"],
                DEFAULT_PARAMS["gamma"],
                DEFAULT_PARAMS["j_n"],
                DEFAULT_PARAMS["j_cross"],
                DEFAULT_PARAMS["i_0"],
                DEFAULT_PARAMS["sigma"],
                DEFAULT_PARAMS["dt"],
                np.zeros(100),
                np.zeros(50),
                np.zeros(200),
            )

    def test_xi_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="xi length must be 2 \\* n_steps"):
            py_wong_wang_simulate(
                0.1,
                0.1,
                DEFAULT_PARAMS["tau_s"],
                DEFAULT_PARAMS["gamma"],
                DEFAULT_PARAMS["j_n"],
                DEFAULT_PARAMS["j_cross"],
                DEFAULT_PARAMS["i_0"],
                DEFAULT_PARAMS["sigma"],
                DEFAULT_PARAMS["dt"],
                np.zeros(100),
                np.zeros(100),
                np.zeros(50),
            )
