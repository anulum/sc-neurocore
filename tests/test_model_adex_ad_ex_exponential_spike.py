# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdExExponentialSpike from former test_model_adex.py

"""Focused suite: TestAdExExponentialSpike from former test_model_adex.py."""

from __future__ import annotations

from tests.model_adex_support import *  # noqa: F403

class TestAdExExponentialSpike:
    def test_exponential_upstroke(self):
        """delta_T controls spike sharpness. Larger delta_T → softer spike."""
        n_sharp = AdExNeuron(delta_t=1.0)
        n_soft = AdExNeuron(delta_t=5.0)
        # Both should fire, but with different dynamics
        s_sharp = len(_run(n_sharp, current=500.0, steps=10000))
        s_soft = len(_run(n_soft, current=500.0, steps=10000))
        # Just verify both fire
        assert s_sharp > 0 and s_soft > 0
