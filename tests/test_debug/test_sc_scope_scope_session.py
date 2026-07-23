# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestScopeSession from former test_sc_scope.py

"""Focused suite: TestScopeSession from former test_sc_scope.py."""

from __future__ import annotations

from sc_scope_support import *  # noqa: F403

class TestScopeSession:
    def _make_session(self, num_layers=2):
        cfg = TransportConfig(TransportType.SIMULATED)
        tb = TransportBackend(cfg)
        la = LiveAnalyzer(num_layers=num_layers)
        return ScopeSession(transport=tb, analyzer=la)

    def test_start_stop(self):
        s = self._make_session()
        assert s.start() is True
        assert s.is_running is True
        s.stop()
        assert s.is_running is False

    def test_capture_one(self):
        s = self._make_session()
        s.start()
        sample = s.capture_one(layer_id=0, num_words=8)
        assert sample is not None
        assert sample.layer_id == 0
        assert s.sample_count == 1
        s.stop()

    def test_capture_sweep(self):
        s = self._make_session(num_layers=3)
        s.start()
        samples = s.capture_sweep(num_layers=3)
        assert len(samples) == 3
        assert s.sample_count == 3
        s.stop()

    def test_error_budget_integration(self):
        s = self._make_session()
        s.add_error_budget(0, expected_density=0.3, tol=0.5)
        s.start()
        for _ in range(10):
            s.capture_one(layer_id=0)
        assert 0 in s.error_budgets
        assert len(s.error_budgets[0].history) == 10
        s.stop()

    def test_status(self):
        s = self._make_session()
        s.start()
        s.capture_one()
        st = s.status()
        assert st["running"] is True
        assert st["samples"] == 1
        assert st["bytes_received"] > 0
        s.stop()

    def test_capture_without_start(self):
        s = self._make_session()
        assert s.capture_one() is None
