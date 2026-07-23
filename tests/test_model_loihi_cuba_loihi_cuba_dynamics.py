# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLoihiCUBADynamics from former test_model_loihi_cuba.py

"""Focused suite: TestLoihiCUBADynamics from former test_model_loihi_cuba.py."""

from __future__ import annotations

from tests.model_loihi_cuba_support import *  # noqa: F403

class TestLoihiCUBADynamics:
    def test_fires(self):
        assert len(_run(LoihiCUBANeuron(), 200, 5000)) >= 50

    def test_rate_monotonic(self):
        s_low = len(_run(LoihiCUBANeuron(), 100, 5000))
        s_high = len(_run(LoihiCUBANeuron(), 500, 5000))
        assert s_high >= s_low

    @pytest.mark.parametrize("current", [50, 100, 200, 500])
    def test_fi_sweep(self, current: int):
        n = LoihiCUBANeuron()
        for _ in range(5000):
            n.step(current)
        assert isinstance(n.v, int)
