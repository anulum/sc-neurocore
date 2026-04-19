# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Parity: Python primary vs Julia simulator (Wong-Wang)

"""Bit-exact parity between the Python primary and the Julia simulator.

The Julia accel kernel consumes the exact same 2-samples-per-step RNG
order as Python's inline `np.random.randn()` pairs, so trajectories are
expected to match to f64 round-off when the caller pre-draws `2 * N`
samples from the same seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.wong_wang import WongWangUnit

pytest.importorskip("juliacall", reason="juliacall not installed")
from sc_neurocore.accel.julia.neurons import simulate_wong_wang  # noqa: E402

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
    np.random.seed(seed)
    u = WongWangUnit(**DEFAULT_PARAMS)
    s1 = np.empty(n_steps, dtype=np.float64)
    s2 = np.empty(n_steps, dtype=np.float64)
    for t in range(n_steps):
        u.step(float(stim1[t]), float(stim2[t]))
        s1[t], s2[t] = u.s1, u.s2
    return s1, s2


def _run_julia(n_steps: int, stim1: np.ndarray, stim2: np.ndarray, seed: int):
    np.random.seed(seed)
    xi = np.random.randn(2 * n_steps).astype(np.float64)
    out = simulate_wong_wang(
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
    return out["s1"], out["s2"]


class TestPythonJuliaParity:
    """Julia simulator trajectory must match Python primary at f64 round-off."""

    def test_parity_quiescent(self):
        n = 5_000
        stim1 = np.zeros(n)
        stim2 = np.zeros(n)
        p1, p2 = _run_python(n, stim1, stim2, seed=42)
        j1, j2 = _run_julia(n, stim1, stim2, seed=42)
        assert np.allclose(p1, j1, atol=1e-12, rtol=0)
        assert np.allclose(p2, j2, atol=1e-12, rtol=0)

    def test_parity_biased(self):
        n = 5_000
        stim1 = np.full(n, 0.1)
        stim2 = np.zeros(n)
        p1, p2 = _run_python(n, stim1, stim2, seed=123)
        j1, j2 = _run_julia(n, stim1, stim2, seed=123)
        assert np.allclose(p1, j1, atol=1e-12, rtol=0)
        assert np.allclose(p2, j2, atol=1e-12, rtol=0)

    def test_parity_across_seeds(self):
        n = 2_000
        stim1 = np.full(n, 0.1)
        stim2 = np.zeros(n)
        for seed in (0, 1, 42, 100, 2025):
            p1, p2 = _run_python(n, stim1, stim2, seed=seed)
            j1, j2 = _run_julia(n, stim1, stim2, seed=seed)
            assert np.allclose(p1, j1, atol=1e-12, rtol=0), f"seed={seed}: s1 diverged"
            assert np.allclose(p2, j2, atol=1e-12, rtol=0), f"seed={seed}: s2 diverged"


class TestRustJuliaCrossParity:
    """Julia + Rust must agree bit-exact when fed the same xi buffer.

    This guards against any drift that would break the multi-backend
    benchmark's "parity" column.
    """

    def test_rust_julia_identical_under_shared_xi(self):
        rs_sim = pytest.importorskip(
            "sc_neurocore_engine", reason="Rust engine not built"
        ).py_wong_wang_simulate
        n = 5_000
        np.random.seed(17)
        xi = np.random.randn(2 * n).astype(np.float64)
        stim1 = np.full(n, 0.15)
        stim2 = np.full(n, 0.05)
        r_out = rs_sim(
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
        j_out = simulate_wong_wang(
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
        # Both batches read the identical xi buffer; difference should be
        # bounded by f64 round-off only.
        assert np.allclose(r_out["s1"], j_out["s1"], atol=1e-12, rtol=0)
        assert np.allclose(r_out["s2"], j_out["s2"], atol=1e-12, rtol=0)
        assert abs(r_out["s1_final"] - j_out["s1_final"]) < 1e-12
        assert abs(r_out["s2_final"] - j_out["s2_final"]) < 1e-12


class TestInputValidation:
    def test_xi_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="xi length must be 2 \\* n_steps"):
            simulate_wong_wang(
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

    def test_stim_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="stim1 and stim2 length mismatch"):
            simulate_wong_wang(
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
