# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Parity: Python primary vs Go simulator (Wong-Wang)

"""Bit-exact parity between the Python primary and the Go cgo simulator."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.wong_wang import WongWangUnit

from sc_neurocore.accel.go.wong_wang import (
    _HAS_GO_WONG_WANG,
    simulate_wong_wang,
)

if not _HAS_GO_WONG_WANG:
    pytest.skip(
        "libwong_wang.so not built (run go build in accel/go/wong_wang)",
        allow_module_level=True,
    )

DEFAULT_PARAMS = dict(
    tau_s=0.1,
    gamma=0.641,
    j_n=0.2609,
    j_cross=0.0497,
    i_0=0.3255,
    sigma=0.02,
    dt=0.001,
)


def _run_python(n: int, stim1, stim2, seed):
    np.random.seed(seed)
    u = WongWangUnit(**DEFAULT_PARAMS)
    s1 = np.empty(n, dtype=np.float64)
    s2 = np.empty(n, dtype=np.float64)
    for t in range(n):
        u.step(float(stim1[t]), float(stim2[t]))
        s1[t], s2[t] = u.s1, u.s2
    return s1, s2


def _run_go(n: int, stim1, stim2, seed):
    np.random.seed(seed)
    xi = np.random.randn(2 * n).astype(np.float64)
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
        np.asarray(stim1, dtype=np.float64),
        np.asarray(stim2, dtype=np.float64),
        xi,
    )
    return out["s1"], out["s2"]


class TestPythonGoParity:
    def test_parity_quiescent(self):
        n = 5_000
        p1, p2 = _run_python(n, np.zeros(n), np.zeros(n), seed=42)
        g1, g2 = _run_go(n, np.zeros(n), np.zeros(n), seed=42)
        assert np.allclose(p1, g1, atol=1e-12, rtol=0)
        assert np.allclose(p2, g2, atol=1e-12, rtol=0)

    def test_parity_biased(self):
        n = 5_000
        stim1 = np.full(n, 0.1)
        stim2 = np.zeros(n)
        p1, p2 = _run_python(n, stim1, stim2, seed=123)
        g1, g2 = _run_go(n, stim1, stim2, seed=123)
        assert np.allclose(p1, g1, atol=1e-12, rtol=0)
        assert np.allclose(p2, g2, atol=1e-12, rtol=0)

    def test_parity_across_seeds(self):
        n = 2_000
        stim1 = np.full(n, 0.1)
        stim2 = np.zeros(n)
        for seed in (0, 1, 42, 100, 2025):
            p1, p2 = _run_python(n, stim1, stim2, seed=seed)
            g1, g2 = _run_go(n, stim1, stim2, seed=seed)
            assert np.allclose(p1, g1, atol=1e-12, rtol=0), f"seed={seed}: diverged"


class TestRustGoCrossParity:
    """Rust + Go must agree bit-exact under shared xi."""

    def test_rust_go_identical(self):
        rs = pytest.importorskip(
            "sc_neurocore_engine", reason="Rust engine not built"
        ).py_wong_wang_simulate
        n = 5_000
        np.random.seed(17)
        xi = np.random.randn(2 * n).astype(np.float64)
        stim1 = np.full(n, 0.15)
        stim2 = np.full(n, 0.05)
        r = rs(
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
        g = simulate_wong_wang(
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
        assert np.allclose(r["s1"], g["s1"], atol=1e-12, rtol=0)
        assert np.allclose(r["s2"], g["s2"], atol=1e-12, rtol=0)


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
