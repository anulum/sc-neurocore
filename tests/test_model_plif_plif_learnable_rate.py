# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPLIFLearnableRate from former test_model_plif.py

"""Focused suite: TestPLIFLearnableRate from former test_model_plif.py."""

from __future__ import annotations

from tests.model_plif_support import *  # noqa: F403


class TestPLIFLearnableRate:
    def test_higher_alpha_more_spikes(self):
        """Higher alpha (more memory) → easier to reach threshold → more spikes."""
        I = 0.4  # Below I_crit for alpha=0.5, above for alpha=0.73
        n_low = ParametricLIFNeuron(a=-1.0)  # alpha ≈ 0.27
        n_high = ParametricLIFNeuron(a=1.0)  # alpha ≈ 0.73
        s_low = sum(n_low.step(I) for _ in range(500))
        s_high = sum(n_high.step(I) for _ in range(500))
        assert s_high > s_low

    @pytest.mark.parametrize("a_val", [-2.0, -1.0, 0.0, 1.0, 2.0])
    def test_rate_at_suprathreshold(self, a_val: float):
        """At I ≥ threshold, rate = 1 spike/step regardless of alpha."""
        n = ParametricLIFNeuron(a=a_val)
        n.step(2.0)  # prime (V=2.0)
        spikes = sum(n.step(2.0) for _ in range(100))
        assert spikes == 100

    def test_subcritical_no_spikes(self):
        """Below I_crit, neuron never fires (voltage converges below threshold)."""
        for a_val in [-2.0, 0.0, 2.0]:
            n = ParametricLIFNeuron(a=a_val)
            alpha = n.alpha
            I_crit = n.threshold * (1.0 - alpha)
            I_test = I_crit * 0.9  # 10% below critical
            spikes = sum(n.step(I_test) for _ in range(1000))
            assert spikes == 0, (
                f"a={a_val}: {spikes} spikes at I={I_test:.4f} < I_crit={I_crit:.4f}"
            )
