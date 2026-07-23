# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMSDynamics from former test_model_mainen_sejnowski.py

"""Focused suite: TestMSDynamics from former test_model_mainen_sejnowski.py."""

from __future__ import annotations

from tests.model_mainen_sejnowski_support import *  # noqa: F403

class TestMSDynamics:
    def test_fires(self):
        n = MainenSejnowskiNeuron()
        spikes = _run(n, current=10.0, steps=500)
        assert len(spikes) >= 1

    def test_rate_monotonic(self):
        s_low = len(_run(MainenSejnowskiNeuron(), 5.0, 500))
        s_high = len(_run(MainenSejnowskiNeuron(), 20.0, 500))
        assert s_high >= s_low

    @pytest.mark.parametrize("current", [0.0, 5.0, 10.0, 20.0])
    def test_fi_sweep(self, current: float):
        n = MainenSejnowskiNeuron()
        for _ in range(200):
            n.step(current)
        assert np.isfinite(n.vs)
