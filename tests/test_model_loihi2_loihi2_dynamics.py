# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLoihi2Dynamics from former test_model_loihi2.py

"""Focused suite: TestLoihi2Dynamics from former test_model_loihi2.py."""

from __future__ import annotations

from tests.model_loihi2_support import *  # noqa: F403

class TestLoihi2Dynamics:
    def test_fires(self):
        assert len(_run(Loihi2Neuron(), 200, 5000)) >= 50

    def test_rate_monotonic(self):
        s_low = len(_run(Loihi2Neuron(), 100, 5000))
        s_high = len(_run(Loihi2Neuron(), 500, 5000))
        assert s_high >= s_low

    @pytest.mark.parametrize("current", [50, 100, 200, 500])
    def test_fi_sweep(self, current: int):
        n = Loihi2Neuron()
        for _ in range(5000):
            n.step(current)
        assert isinstance(n.s1, int)
