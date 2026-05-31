# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Parity: Python primary vs Rust simulator (Wilson-Cowan)

"""Last-ulp parity between `WilsonCowanUnit.step` and the Rust
`py_wilson_cowan_simulate`.

Wilson-Cowan is deterministic (no stochastic noise), so Python and
Rust must produce numerically identical trajectories up to the
last few IEEE-754 ulps for any external-input sequence.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.wilson_cowan import WilsonCowanUnit

pytest.importorskip(
    "sc_neurocore_engine", reason="Rust engine wheel not installed (maturin develop)"
)
from sc_neurocore_engine import py_wilson_cowan_simulate  # noqa: E402


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


def _run_rust(ext: np.ndarray):
    out = py_wilson_cowan_simulate(
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


class TestBitExactParity:
    """Wilson-Cowan is noise-free; Python and Rust traces must match within
    a small last-ulp envelope across compiler and Python-version math paths."""

    def test_parity_zero_input(self):
        n = 5_000
        e_py, i_py = _run_python(np.zeros(n))
        e_rs, i_rs = _run_rust(np.zeros(n))
        assert np.allclose(e_py, e_rs, atol=2e-14, rtol=0)
        assert np.allclose(i_py, i_rs, atol=2e-14, rtol=0)

    def test_parity_constant_drive(self):
        n = 5_000
        ext = np.full(n, 1.5)
        e_py, i_py = _run_python(ext)
        e_rs, i_rs = _run_rust(ext)
        assert np.allclose(e_py, e_rs, atol=2e-14, rtol=0)
        assert np.allclose(i_py, i_rs, atol=2e-14, rtol=0)

    def test_parity_time_varying_drive(self):
        n = 3_000
        ext = np.sin(np.linspace(0, 10 * np.pi, n)) * 2.0
        e_py, i_py = _run_python(ext)
        e_rs, i_rs = _run_rust(ext)
        assert np.allclose(e_py, e_rs, atol=2e-14, rtol=0)
        assert np.allclose(i_py, i_rs, atol=2e-14, rtol=0)

    def test_parity_step_function_drive(self):
        """Sharp transitions are the hardest test for integration parity."""
        n = 4_000
        ext = np.zeros(n)
        ext[1_000:2_000] = 5.0
        ext[3_000:] = -2.0
        e_py, i_py = _run_python(ext)
        e_rs, i_rs = _run_rust(ext)
        assert np.allclose(e_py, e_rs, atol=2e-14, rtol=0)
        assert np.allclose(i_py, i_rs, atol=2e-14, rtol=0)


class TestFinalStateParity:
    def test_rust_e_final_matches_trace_last(self):
        n = 2_000
        ext = np.full(n, 1.0)
        out = py_wilson_cowan_simulate(
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
        assert out["e_final"] == out["e"][-1]
        assert out["i_final"] == out["i"][-1]

    def test_python_rust_final_state_match(self):
        n = 3_000
        ext = np.full(n, 2.5)
        e_py, i_py = _run_python(ext)
        e_rs, i_rs = _run_rust(ext)
        assert abs(e_py[-1] - e_rs[-1]) < 2e-14
        assert abs(i_py[-1] - i_rs[-1]) < 2e-14


class TestEdgeCases:
    def test_zero_length_workload(self):
        out = py_wilson_cowan_simulate(
            0.3,
            0.2,
            10.0,
            6.0,
            10.0,
            1.0,
            1.0,
            2.0,
            1.2,
            4.0,
            0.1,
            np.zeros(0),
        )
        assert out["e"].shape == (0,)
        assert out["e_final"] == 0.3
        assert out["i_final"] == 0.2

    def test_single_step(self):
        out = py_wilson_cowan_simulate(
            0.1,
            0.05,
            10.0,
            6.0,
            10.0,
            1.0,
            1.0,
            2.0,
            1.2,
            4.0,
            0.1,
            np.array([1.0]),
        )
        assert out["e"].shape == (1,)
        assert out["e_final"] == out["e"][0]
