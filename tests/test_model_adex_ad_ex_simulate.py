# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdExSimulate from former test_model_adex.py

"""Focused suite: TestAdExSimulate from former test_model_adex.py."""

from __future__ import annotations

from tests.model_adex_support import *  # noqa: F403

class TestAdExSimulate:
    """Engineering-verification surface for ``AdExNeuron.simulate``."""

    def test_simulate_python_returns_finite_trace_and_spike_count(self) -> None:
        n = AdExNeuron()
        trace, spikes = n.simulate(1000, current=250.0, backend="python")
        assert trace.shape == (1000,)
        assert np.all(np.isfinite(trace))
        assert spikes >= 1
        assert n.v == float(trace[-1])

    def test_simulate_rejects_negative_steps_and_bad_backend(self) -> None:
        n = AdExNeuron()
        with pytest.raises(ValueError, match="n_steps"):
            n.simulate(-1, current=0.0, backend="python")
        with pytest.raises(ValueError, match="backend"):
            n.simulate(10, current=0.0, backend="cuda")
        with pytest.raises(ValueError, match="current"):
            n.simulate(10, current=float("nan"), backend="python")

    def test_simulate_rust_matches_python_under_default_contract(self) -> None:
        pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
        py = AdExNeuron()
        rs = AdExNeuron()
        tr_py, sp_py = py.simulate(1000, current=250.0, backend="python")
        tr_rs, sp_rs = rs.simulate(1000, current=250.0, backend="rust")
        assert sp_py == sp_rs
        assert np.array_equal(tr_py, tr_rs)
        assert (rs.v, rs.w) == (py.v, py.w)

    def test_simulate_rust_rejects_non_default_contract(self) -> None:
        pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
        n = AdExNeuron(integrator="rk4")
        with pytest.raises(RuntimeError, match="factory-default"):
            n.simulate(10, current=0.0, backend="rust")
        n2 = AdExNeuron(v=-70.0)
        with pytest.raises(RuntimeError, match="factory-default"):
            n2.simulate(10, current=0.0, backend="rust")

    def test_simulate_zero_steps_is_empty(self) -> None:
        n = AdExNeuron()
        before = (n.v, n.w)
        trace, spikes = n.simulate(0, current=250.0, backend="python")
        assert trace.shape == (0,)
        assert spikes == 0
        assert (n.v, n.w) == before
