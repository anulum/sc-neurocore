# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Rust engine integration in Studio

from __future__ import annotations

import numpy as np
import pytest

try:
    from sc_neurocore_engine.sc_neurocore_engine import (  # noqa: F401
        py_simulate_ei_network as _ei_check,
        py_batch_simulate as _batch_check,
    )

    _RUST = True
except (ImportError, ModuleNotFoundError):
    _RUST = False

needs_rust = pytest.mark.skipif(not _RUST, reason="Rust engine pyd not loadable")


class TestRustEINetwork:
    @needs_rust
    def test_rust_ei_network_direct(self):
        from sc_neurocore_engine.sc_neurocore_engine import py_simulate_ei_network

        r = py_simulate_ei_network(n_exc=20, n_inh=5, duration=50.0, ext_rate=100.0)
        assert int(r["n_total"]) == 25
        assert int(r["n_exc"]) == 20
        assert "spike_times" in r
        assert "exc_rates" in r

    @needs_rust
    def test_rust_ei_network_via_python(self):
        from sc_neurocore.studio.network import simulate_ei_network

        r = simulate_ei_network(n_exc=20, n_inh=5, duration=50.0, ext_rate=100.0)
        assert r["n_total"] == 25
        assert isinstance(r["spike_times"], list)
        assert isinstance(r["mean_exc_rate"], float)

    @needs_rust
    def test_rust_ei_result_types(self):
        from sc_neurocore_engine.sc_neurocore_engine import py_simulate_ei_network

        r = py_simulate_ei_network(n_exc=10, n_inh=5, duration=20.0)
        assert hasattr(r["spike_times"], "tolist")
        assert hasattr(r["rate_time"], "tolist")

    def test_numpy_fallback_works(self):
        from sc_neurocore.studio.network import _simulate_numpy

        r = _simulate_numpy(
            n_exc=10,
            n_inh=5,
            w_ee=0.1,
            w_ei=0.4,
            w_ie=0.1,
            w_ii=0.4,
            p_conn=0.2,
            ext_rate=10.0,
            duration=20.0,
            dt=0.1,
        )
        assert r["n_total"] == 15
        assert isinstance(r["spike_times"], list)


class TestRustBatchSimulate:
    @needs_rust
    def test_batch_adex(self):
        from sc_neurocore_engine.sc_neurocore_engine import py_batch_simulate

        current = np.full(1000, 500.0)
        r = py_batch_simulate("AdEx", 1000, current)
        assert len(r["voltages"]) == 1000
        assert all(np.isfinite(r["voltages"]))

    @needs_rust
    def test_batch_hodgkin_huxley(self):
        from sc_neurocore_engine.sc_neurocore_engine import py_batch_simulate

        current = np.full(500, 15.0)
        r = py_batch_simulate("HodgkinHuxley", 500, current)
        assert len(r["voltages"]) == 500
        assert r["n_steps"] == 500

    @needs_rust
    def test_batch_izhikevich(self):
        from sc_neurocore_engine.sc_neurocore_engine import py_batch_simulate

        current = np.full(1000, 10.0)
        r = py_batch_simulate("Izhikevich", 1000, current)
        spikes = r["spikes"].tolist()
        assert len(spikes) > 0, "Izhikevich with I=10 should spike"

    @needs_rust
    def test_batch_unsupported_model(self):
        from sc_neurocore_engine.sc_neurocore_engine import py_batch_simulate

        current = np.full(100, 10.0)
        with pytest.raises(Exception):
            py_batch_simulate("NonexistentModel", 100, current)

    @needs_rust
    def test_model_simulate_uses_rust(self):
        from sc_neurocore.studio.models import _try_rust_simulate
        from sc_neurocore.studio.simulation import _make_current_trace

        I = _make_current_trace("constant", 500.0, 1000)
        result = _try_rust_simulate("AdEx", 1000, I, 0.1)
        assert result is not None, "Rust path should succeed for AdEx"
        assert "states" in result
        assert "v" in result["states"]
