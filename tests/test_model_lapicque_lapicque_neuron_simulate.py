# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLapicqueNeuronSimulate from former test_model_lapicque.py

"""Focused suite: TestLapicqueNeuronSimulate from former test_model_lapicque.py."""

from __future__ import annotations

from tests.model_lapicque_support import *  # noqa: F403


class TestLapicqueNeuronSimulate:
    """Engineering-verification surface for ``LapicqueNeuron.simulate``."""

    def test_simulate_python_returns_finite_trace(self) -> None:
        n = LapicqueNeuron()
        trace, spikes = n.simulate(1000, current=2.0, backend="python")
        assert trace.shape == (1000,)
        assert np.all(np.isfinite(trace))
        assert spikes >= 1

    def test_simulate_rust_matches_or_ulp_python(self) -> None:
        pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
        py = LapicqueNeuron()
        rs = LapicqueNeuron()
        tr_py, sp_py = py.simulate(1000, current=2.0, backend="python")
        tr_rs, sp_rs = rs.simulate(1000, current=2.0, backend="rust")
        assert sp_py == sp_rs
        max_diff = float(np.max(np.abs(tr_py - tr_rs)))
        assert max_diff < 1e-9

    def test_simulate_rust_carries_non_default_complete_contract(self) -> None:
        pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
        python = LapicqueNeuron(v=0.2, tau=7.0, resistance=1.5, dt=0.02)
        rust = LapicqueNeuron(v=0.2, tau=7.0, resistance=1.5, dt=0.02)
        expected_trace, expected_events = python.simulate_complete(100, 2.0, backend="python")
        trace, events = rust.simulate_complete(100, 2.0, backend="rust")
        np.testing.assert_allclose(trace, expected_trace, atol=2.0e-15, rtol=0.0)
        np.testing.assert_array_equal(events, expected_events)
        assert rust.v == pytest.approx(python.v, abs=2.0e-15)
