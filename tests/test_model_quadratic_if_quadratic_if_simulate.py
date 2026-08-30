# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQuadraticIFSimulate from former test_model_quadratic_if.py

"""Focused suite: TestQuadraticIFSimulate from former test_model_quadratic_if.py."""

from __future__ import annotations

from tests.model_quadratic_if_support import *  # noqa: F403


class TestQuadraticIFSimulate:
    """Engineering-verification surface for ``QuadraticIFNeuron.simulate``."""

    def test_simulate_python_returns_finite_trace(self) -> None:
        n = QuadraticIFNeuron()
        trace, spikes = n.simulate(1000, current=1.0, backend="python")
        assert trace.shape == (1000,)
        assert np.all(np.isfinite(trace))
        assert spikes >= 1
        assert n.v == float(trace[-1])

    def test_simulate_rust_matches_python(self) -> None:
        pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
        py = QuadraticIFNeuron()
        rs = QuadraticIFNeuron()
        tr_py, sp_py = py.simulate(1000, current=1.0, backend="python")
        tr_rs, sp_rs = rs.simulate(1000, current=1.0, backend="rust")
        assert sp_py == sp_rs
        assert np.array_equal(tr_py, tr_rs)

    def test_simulate_rust_accepts_non_default_complete_contract(self) -> None:
        pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
        py = QuadraticIFNeuron(v=-0.5, v_reset=-2.0, v_peak=2.0, dt=0.02)
        rs = QuadraticIFNeuron(v=-0.5, v_reset=-2.0, v_peak=2.0, dt=0.02)
        py_voltage, py_events = py.simulate_complete(100, current=1.5, backend="python")
        rs_voltage, rs_events = rs.simulate_complete(100, current=1.5, backend="rust")
        np.testing.assert_allclose(rs_voltage, py_voltage, rtol=0.0, atol=2.0e-12)
        np.testing.assert_array_equal(rs_events, py_events)
        assert rs.v == pytest.approx(py.v, abs=2.0e-12)
