# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Parity: Python primary vs Julia simulator (Wilson-Cowan)

"""Bit-exact parity between the Python primary and the Julia N-step
simulator for the Wilson-Cowan E/I model.

Wilson-Cowan is deterministic (no stochastic noise); both backends
evaluate the same sigmoid + forward-Euler arithmetic in f64, so
trajectories match at machine-epsilon.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.wilson_cowan import WilsonCowanUnit

pytest.importorskip("juliacall", reason="juliacall not installed")
from sc_neurocore.accel.julia.neurons import simulate_wilson_cowan  # noqa: E402

DEFAULT_PARAMS = dict(
    w_ee=10.0,
    w_ei=6.0,
    w_ie=10.0,
    w_ii=1.0,
    tau_e=1.0,
    tau_i=2.0,
    a=1.2,
    theta=4.0,
    dt=0.1,
)


def _run_python(ext: np.ndarray):
    u = WilsonCowanUnit(**DEFAULT_PARAMS)
    e = np.empty(ext.size, dtype=np.float64)
    i = np.empty(ext.size, dtype=np.float64)
    for t in range(ext.size):
        u.step(float(ext[t]))
        e[t], i[t] = u.e, u.i
    return e, i


def _run_julia(ext: np.ndarray):
    out = simulate_wilson_cowan(
        0.1,
        0.05,
        DEFAULT_PARAMS["w_ee"],
        DEFAULT_PARAMS["w_ei"],
        DEFAULT_PARAMS["w_ie"],
        DEFAULT_PARAMS["w_ii"],
        DEFAULT_PARAMS["tau_e"],
        DEFAULT_PARAMS["tau_i"],
        DEFAULT_PARAMS["a"],
        DEFAULT_PARAMS["theta"],
        DEFAULT_PARAMS["dt"],
        ext.astype(np.float64),
    )
    return out["e"], out["i"]


class TestPythonJuliaParity:
    """Julia trajectory must match Python primary at machine-epsilon."""

    def test_parity_zero_input(self):
        n = 3_000
        e_py, i_py = _run_python(np.zeros(n))
        e_jl, i_jl = _run_julia(np.zeros(n))
        assert np.allclose(e_py, e_jl, atol=1e-14, rtol=0)
        assert np.allclose(i_py, i_jl, atol=1e-14, rtol=0)

    def test_parity_constant_drive(self):
        n = 3_000
        ext = np.full(n, 1.5)
        e_py, i_py = _run_python(ext)
        e_jl, i_jl = _run_julia(ext)
        assert np.allclose(e_py, e_jl, atol=1e-14, rtol=0)
        assert np.allclose(i_py, i_jl, atol=1e-14, rtol=0)

    def test_parity_sinusoid_drive(self):
        n = 2_000
        ext = np.sin(np.linspace(0, 8 * np.pi, n)) * 2.0
        e_py, i_py = _run_python(ext)
        e_jl, i_jl = _run_julia(ext)
        assert np.allclose(e_py, e_jl, atol=1e-14, rtol=0)
        assert np.allclose(i_py, i_jl, atol=1e-14, rtol=0)


class TestRustJuliaCrossParity:
    """Rust + Julia must agree to machine epsilon under identical inputs."""

    def test_rust_julia_identical(self):
        rs = pytest.importorskip(
            "sc_neurocore_engine", reason="Rust engine required"
        ).py_wilson_cowan_simulate
        n = 3_000
        ext = np.linspace(-1.0, 4.0, n)
        r_out = rs(
            0.1,
            0.05,
            DEFAULT_PARAMS["w_ee"],
            DEFAULT_PARAMS["w_ei"],
            DEFAULT_PARAMS["w_ie"],
            DEFAULT_PARAMS["w_ii"],
            DEFAULT_PARAMS["tau_e"],
            DEFAULT_PARAMS["tau_i"],
            DEFAULT_PARAMS["a"],
            DEFAULT_PARAMS["theta"],
            DEFAULT_PARAMS["dt"],
            ext,
        )
        j_out = simulate_wilson_cowan(
            0.1,
            0.05,
            DEFAULT_PARAMS["w_ee"],
            DEFAULT_PARAMS["w_ei"],
            DEFAULT_PARAMS["w_ie"],
            DEFAULT_PARAMS["w_ii"],
            DEFAULT_PARAMS["tau_e"],
            DEFAULT_PARAMS["tau_i"],
            DEFAULT_PARAMS["a"],
            DEFAULT_PARAMS["theta"],
            DEFAULT_PARAMS["dt"],
            ext,
        )
        assert np.allclose(r_out["e"], j_out["e"], atol=1e-14, rtol=0)
        assert np.allclose(r_out["i"], j_out["i"], atol=1e-14, rtol=0)
        assert abs(r_out["e_final"] - j_out["e_final"]) < 1e-14
        assert abs(r_out["i_final"] - j_out["i_final"]) < 1e-14
