# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEventSynchronization from former test_spike_train_stats.py

"""Focused suite: TestEventSynchronization from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403


class TestEventSynchronization:
    def test_identical_high(self):
        train = _poisson_train(50.0, 0.5)
        s = event_synchronization(train, train, tau_ms=2.0)
        assert s > 0.5

    def test_independent_low(self):
        a = _poisson_train(50.0, 0.5, seed=1)
        b = _poisson_train(50.0, 0.5, seed=2)
        s = event_synchronization(a, b, tau_ms=1.0)
        assert s < 0.5

    def test_pure_python_fallback_when_rust_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force the pure-Python double-loop branch; the native accelerator is
        # built in this environment and would otherwise shadow it.
        monkeypatch.setattr("sc_neurocore.analysis.spike_stats.correlation._HAS_RUST", False)
        train = _poisson_train(50.0, 0.5)
        s = event_synchronization(train, train, tau_ms=2.0)
        assert s > 0.5
