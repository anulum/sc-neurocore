# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Parity: Python primary vs Go simulator (Wilson-Cowan)

"""Bit-exact parity between the Python primary and the Go cgo
N-step simulator for the Wilson-Cowan E/I model."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.wilson_cowan import WilsonCowanUnit

from sc_neurocore.accel.go.wilson_cowan import (
    _HAS_GO_WILSON_COWAN,
    simulate_wilson_cowan,
)

if not _HAS_GO_WILSON_COWAN:
    pytest.skip(
        "libwilson_cowan.so not built (run go build in accel/go/wilson_cowan)",
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


def _run_go(ext: np.ndarray):
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


class TestPythonGoParity:
    def test_parity_zero_input(self):
        n = 3_000
        e_py, i_py = _run_python(np.zeros(n))
        e_go, i_go = _run_go(np.zeros(n))
        assert np.allclose(e_py, e_go, atol=1e-14, rtol=0)
        assert np.allclose(i_py, i_go, atol=1e-14, rtol=0)

    def test_parity_constant_drive(self):
        n = 3_000
        ext = np.full(n, 2.0)
        e_py, i_py = _run_python(ext)
        e_go, i_go = _run_go(ext)
        assert np.allclose(e_py, e_go, atol=1e-14, rtol=0)
        assert np.allclose(i_py, i_go, atol=1e-14, rtol=0)

    def test_parity_ramp_drive(self):
        n = 3_000
        ext = np.linspace(-1.0, 4.0, n)
        e_py, i_py = _run_python(ext)
        e_go, i_go = _run_go(ext)
        assert np.allclose(e_py, e_go, atol=1e-14, rtol=0)
        assert np.allclose(i_py, i_go, atol=1e-14, rtol=0)


class TestRustGoCrossParity:
    def test_rust_go_identical(self):
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
        g_out = simulate_wilson_cowan(
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
        assert np.allclose(r_out["e"], g_out["e"], atol=1e-14, rtol=0)
        assert np.allclose(r_out["i"], g_out["i"], atol=1e-14, rtol=0)
