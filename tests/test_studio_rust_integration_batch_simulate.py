# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio rust integration batch simulate

"""Focused suite: TestRustBatchSimulate from former test_studio_rust_integration.py."""

from __future__ import annotations

from tests.studio_rust_integration_support import *  # noqa: F403

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

