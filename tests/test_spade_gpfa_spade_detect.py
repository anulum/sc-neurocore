# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpadeDetect from former test_spade_gpfa.py

"""Focused suite: TestSpadeDetect from former test_spade_gpfa.py."""

from __future__ import annotations

from tests.spade_gpfa_support import *  # noqa: F403


class TestSpadeDetect:
    def test_planted_pattern_detected(self):
        trains = _sync_trains(n_neurons=5, n_steps=2000, sync_every=100)
        results = spade_detect(
            trains,
            bin_ms=5.0,
            dt=0.001,
            min_support=3,
            n_surrogates=50,
            alpha=0.05,
            seed=42,
        )
        detected_sets = [frozenset(r["neurons"]) for r in results]
        assert any(frozenset([0, 1, 2]).issubset(s) for s in detected_sets)

    def test_empty_trains(self):
        assert spade_detect([], bin_ms=5.0) == []

    def test_single_train(self):
        trains = [np.zeros(100, dtype=np.uint8)]
        assert spade_detect(trains) == []

    def test_results_have_required_keys(self):
        trains = _sync_trains()
        results = spade_detect(trains, n_surrogates=20, alpha=0.5)
        for r in results:
            assert "neurons" in r
            assert "lags" in r
            assert "count" in r
            assert "p_value" in r
            assert 0.0 <= r["p_value"] <= 1.0

    def test_no_false_positives_on_independent(self):
        rng = np.random.default_rng(99)
        trains = []
        for i in range(4):
            t = np.zeros(500, dtype=np.uint8)
            t[rng.choice(500, size=5, replace=False)] = 1
            trains.append(t)
        results = spade_detect(trains, min_support=3, n_surrogates=50, alpha=0.01)
        assert len(results) == 0
