# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Parity: Python primary vs Mojo simulator (Wong-Wang)

"""Parity between the Python primary and the Mojo simulator.

Mojo's `exp` comes from the system libm, Rust's comes from Rust std's
`f64::exp`; both are IEEE-compliant but the last-ulp bit pattern can
differ on some inputs. Tolerance is set to 1e-9 (comfortably inside
the physical regime of interest) with a secondary 1e-13 check on the
first 1 000 steps where drift has not yet amplified.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.wong_wang import WongWangUnit

from sc_neurocore.accel.mojo.wong_wang import (
    _HAS_MOJO_WONG_WANG,
    simulate_wong_wang,
)

if not _HAS_MOJO_WONG_WANG:
    pytest.skip(
        "libwong_wang.so not built (run mojo build in accel/mojo/wong_wang)",
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


def _run_mojo(n: int, stim1, stim2, seed):
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


class TestPythonMojoParity:
    def test_parity_biased(self):
        n = 5_000
        stim1 = np.full(n, 0.1)
        stim2 = np.zeros(n)
        p1, p2 = _run_python(n, stim1, stim2, seed=42)
        m1, m2 = _run_mojo(n, stim1, stim2, seed=42)
        # Early trajectory: bit-level parity
        assert np.allclose(p1[:1_000], m1[:1_000], atol=1e-13, rtol=0)
        # Full trajectory: numerical parity within libm vs f64::exp drift
        assert np.allclose(p1, m1, atol=1e-9, rtol=0)
        assert np.allclose(p2, m2, atol=1e-9, rtol=0)

    def test_parity_quiescent(self):
        n = 3_000
        p1, p2 = _run_python(n, np.zeros(n), np.zeros(n), seed=7)
        m1, m2 = _run_mojo(n, np.zeros(n), np.zeros(n), seed=7)
        assert np.allclose(p1, m1, atol=1e-9, rtol=0)
        assert np.allclose(p2, m2, atol=1e-9, rtol=0)


class TestRustMojoCrossParity:
    """Rust + Mojo must agree within libm-vs-f64::exp ulp drift."""

    def test_rust_mojo_within_ulp_drift(self):
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
        m = simulate_wong_wang(
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
        # libm exp and Rust f64::exp can differ by ~1e-14 per call; over
        # 5 000 non-linear iterations that leaves well under 1e-9.
        assert np.allclose(r["s1"], m["s1"], atol=1e-9, rtol=0)
        assert np.allclose(r["s2"], m["s2"], atol=1e-9, rtol=0)


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
