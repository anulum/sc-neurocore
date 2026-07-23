# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMetrics from former test_reservoir.py

"""Focused suite: TestMetrics from former test_reservoir.py."""

from __future__ import annotations

from tests.reservoir_support import *  # noqa: F403

class TestMetrics:
    def test_metrics_returns_all_fields(self):
        res = AutoCriticalReservoir(n_inputs=1, n_neurons=50, seed=0)
        inputs = np.random.randn(30, 1)
        m = res.metrics(inputs)
        assert isinstance(m, ReservoirMetrics)
        assert 0.0 <= m.firing_fraction <= 1.0
        assert m.criticality_error >= 0.0
        assert 0.0 <= m.kernel_quality <= 1.0
        assert m.spectral_radius > 0

    def test_metrics_summary_string(self):
        m = ReservoirMetrics(
            firing_fraction=0.48,
            criticality_error=0.02,
            kernel_quality=0.95,
            spectral_radius=1.1,
        )
        s = m.summary()
        assert "firing=0.480" in s
        assert "spectral_r=1.100" in s
