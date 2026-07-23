# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Rust engine integration in Studio

from __future__ import annotations

import importlib
import numpy as np
import pytest

from sc_neurocore_engine.studio import get_ei_network_simulator


def _bridge_engine():
    """Import the engine through the maintained bridge surface."""
    mod = pytest.importorskip("sc_neurocore_engine")
    if not hasattr(mod, "py_simulate_ei_network"):
        pytest.skip("Rust engine bridge missing Studio functions")
    return mod


def _inner_engine():
    """Import the inner extension module without bypassing the package bridge."""
    return importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


class TestRustEINetwork:
    def test_rust_ei_network_direct(self):
        _bridge_engine()
        simulate = get_ei_network_simulator()
        r = simulate(n_exc=20, n_inh=5, duration=50.0, ext_rate=100.0)
        assert int(r["n_total"]) == 25
        assert int(r["n_exc"]) == 20
        assert "spike_times" in r
        assert "exc_rates" in r

    def test_rust_ei_network_via_python(self):
        _bridge_engine()
        from sc_neurocore.studio.network import simulate_ei_network

        r = simulate_ei_network(n_exc=20, n_inh=5, duration=50.0, ext_rate=100.0)
        assert r["n_total"] == 25
        assert isinstance(r["spike_times"], list)
        assert isinstance(r["mean_exc_rate"], (int, float))

    def test_rust_ei_result_types(self):
        sce = _inner_engine()
        r = sce.py_simulate_ei_network(n_exc=10, n_inh=5, duration=20.0)
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
    def test_batch_adex(self):
        sce = _inner_engine()
        current = np.full(1000, 500.0)
        r = sce.py_batch_simulate("AdEx", 1000, current)
        assert len(r["voltages"]) == 1000
        assert all(np.isfinite(r["voltages"]))

    def test_batch_hodgkin_huxley(self):
        sce = _inner_engine()
        current = np.full(500, 15.0)
        r = sce.py_batch_simulate("HodgkinHuxley", 500, current)
        assert len(r["voltages"]) == 500
        assert r["n_steps"] == 500

    def test_batch_izhikevich(self):
        sce = _inner_engine()
        current = np.full(1000, 10.0)
        r = sce.py_batch_simulate("Izhikevich", 1000, current)
        spikes = r["spikes"].tolist()
        assert len(spikes) > 0, "Izhikevich with I=10 should spike"

    def test_batch_unsupported_model(self):
        sce = _inner_engine()
        current = np.full(100, 10.0)
        with pytest.raises(Exception):
            sce.py_batch_simulate("NonexistentModel", 100, current)

    def test_model_simulate_uses_rust(self):
        """Rust fast path in simulate_model must use the maintained bridge."""
        _bridge_engine()
        from sc_neurocore.studio.models import _try_rust_simulate
        from sc_neurocore.studio.simulation import _make_current_trace

        I = _make_current_trace("constant", 500.0, 1000)
        result = _try_rust_simulate("AdEx", 1000, I, 0.1)
        assert result is not None
        assert "states" in result
        assert "v" in result["states"]

    def test_model_simulate_returns_none_when_backend_unavailable(self, monkeypatch):
        import sc_neurocore.studio.model_simulate as simulate_mod
        import sc_neurocore.studio.models as mod

        def _unavailable():
            raise mod.RustStudioBackendUnavailable("no bridge")

        # Patch the defining module — ``models`` is a thin re-export facade.
        monkeypatch.setattr(simulate_mod, "_load_rust_batch_simulate", _unavailable)
        current = np.full(10, 1.0)
        assert mod._try_rust_simulate("AdEx", 10, current, 0.1) is None

    def test_model_simulate_raises_on_runtime_backend_failure(self, monkeypatch):
        import sc_neurocore.studio.model_simulate as simulate_mod
        import sc_neurocore.studio.models as mod

        def _broken(_name, _n_steps, _current):
            raise RuntimeError("ffi exploded")

        monkeypatch.setattr(simulate_mod, "_load_rust_batch_simulate", lambda: _broken)
        current = np.full(10, 1.0)
        with pytest.raises(mod.RustStudioBackendError, match="AdEx"):
            mod._try_rust_simulate("AdEx", 10, current, 0.1)
