# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCostFunctions from former test_autofit.py

"""Focused suite: TestCostFunctions from former test_autofit.py."""

from __future__ import annotations

from tests.autofit_support import *  # noqa: F403

class TestCostFunctions:
    def test_rmse_identical(self):
        v = np.random.randn(100)
        assert _cost_rmse(v, v) == pytest.approx(0.0)

    def test_rmse_different(self):
        a = np.zeros(100)
        b = np.ones(100)
        assert _cost_rmse(a, b) == pytest.approx(1.0)

    def test_rmse_different_lengths(self):
        a = np.zeros(100)
        b = np.ones(50)
        result = _cost_rmse(a, b)
        assert result == pytest.approx(1.0)

    def test_feature_cost_identical(self):
        feats = {"spike_count": 10, "mean_isi": 5.0, "v_rest": -70.0, "v_max": 30.0, "v_min": -80.0}
        cost = _cost_features(feats, feats)
        assert cost == pytest.approx(0.0, abs=1e-6)

    def test_feature_cost_different_spike_count(self):
        target = {
            "spike_count": 10,
            "mean_isi": 5.0,
            "v_rest": -70.0,
            "v_max": 30.0,
            "v_min": -80.0,
        }
        model = {"spike_count": 5, "mean_isi": 5.0, "v_rest": -70.0, "v_max": 30.0, "v_min": -80.0}
        cost = _cost_features(target, model)
        assert cost > 0

    def test_feature_cost_no_spikes(self):
        target = {
            "spike_count": 0,
            "mean_isi": 0.0,
            "v_rest": -70.0,
            "v_max": -60.0,
            "v_min": -80.0,
        }
        model = {"spike_count": 5, "mean_isi": 3.0, "v_rest": -70.0, "v_max": -60.0, "v_min": -80.0}
        cost = _cost_features(target, model)
        assert cost > 0
