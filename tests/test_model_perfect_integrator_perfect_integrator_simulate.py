# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPerfectIntegratorSimulate from former test_model_perfect_integrator.py

"""Focused suite: TestPerfectIntegratorSimulate from former test_model_perfect_integrator.py."""

from __future__ import annotations

from tests.model_perfect_integrator_support import *  # noqa: F403


class TestPerfectIntegratorSimulate:
    """Engineering-verification surface for ``PerfectIntegratorNeuron.simulate``."""

    def test_simulate_python_returns_finite_trace(self) -> None:
        n = PerfectIntegratorNeuron()
        trace, spikes = n.simulate(1000, current=1.0, backend="python")
        assert trace.shape == (1000,)
        assert np.all(np.isfinite(trace))
        assert spikes == 90
        assert n.v == float(trace[-1])

    def test_simulate_rust_matches_python(self) -> None:
        pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
        py = PerfectIntegratorNeuron()
        rs = PerfectIntegratorNeuron()
        tr_py, sp_py = py.simulate(1000, current=1.0, backend="python")
        tr_rs, sp_rs = rs.simulate(1000, current=1.0, backend="rust")
        assert sp_py == sp_rs
        assert np.array_equal(tr_py, tr_rs)

    def test_simulate_rust_carries_non_default_contract(self) -> None:
        pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
        py = PerfectIntegratorNeuron(c_m=2.0, v=0.25)
        rust = PerfectIntegratorNeuron(c_m=2.0, v=0.25)
        expected = py.simulate(10, current=1.0, backend="python")
        actual = rust.simulate(10, current=1.0, backend="rust")
        assert np.array_equal(actual[0], expected[0])
        assert actual[1] == expected[1]
        assert rust.v == py.v
