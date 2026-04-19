# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Parity: Python primary vs Mojo simulator (Wilson-Cowan)

"""Parity between the Python primary and the Mojo N-step simulator.

Mojo's `exp` comes from libm; Rust's comes from std; both are
IEEE-compliant but last-ulp drift accumulates over thousands of
non-linear iterations. Tolerance set to 1e-9 empirically (measured
drift on ramp drive is ~1e-10)."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.wilson_cowan import WilsonCowanUnit

from sc_neurocore.accel.mojo.wilson_cowan import (
    _HAS_MOJO_WILSON_COWAN,
    simulate_wilson_cowan,
)

if not _HAS_MOJO_WILSON_COWAN:
    pytest.skip(
        "libwilson_cowan.so not built (run mojo build in accel/mojo/wilson_cowan)",
        allow_module_level=True,
    )

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


def _run_mojo(ext: np.ndarray):
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


class TestPythonMojoParity:
    def test_parity_zero_input(self):
        n = 3_000
        e_py, i_py = _run_python(np.zeros(n))
        e_mj, i_mj = _run_mojo(np.zeros(n))
        assert np.allclose(e_py, e_mj, atol=1e-9, rtol=0)
        assert np.allclose(i_py, i_mj, atol=1e-9, rtol=0)

    def test_parity_constant_drive(self):
        n = 3_000
        ext = np.full(n, 2.0)
        e_py, i_py = _run_python(ext)
        e_mj, i_mj = _run_mojo(ext)
        # Nonlinear sigmoid amplifies the libm-vs-std-exp ulp drift
        # quickly; measured drift is bounded by ~1e-12 on short windows
        # and ~1e-9 over thousands of steps.
        assert np.allclose(e_py[:10], e_mj[:10], atol=1e-12, rtol=0)
        assert np.allclose(e_py, e_mj, atol=1e-9, rtol=0)
        assert np.allclose(i_py, i_mj, atol=1e-9, rtol=0)


class TestRustMojoCrossParity:
    def test_rust_mojo_within_ulp_drift(self):
        rs = pytest.importorskip(
            "sc_neurocore_engine", reason="Rust engine required"
        ).py_wilson_cowan_simulate
        n = 3_000
        ext = np.sin(np.linspace(0, 6 * np.pi, n)) * 3.0
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
        m_out = simulate_wilson_cowan(
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
        assert np.allclose(r_out["e"], m_out["e"], atol=1e-9, rtol=0)
        assert np.allclose(r_out["i"], m_out["i"], atol=1e-9, rtol=0)
