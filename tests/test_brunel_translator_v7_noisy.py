# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestV7Noisy from former test_brunel_translator.py

"""Focused suite: TestV7Noisy from former test_brunel_translator.py."""

from __future__ import annotations

from tests.brunel_translator_support import *  # noqa: F403

class TestV7Noisy:
    """V7: Noisy LIF fires stochastically."""

    def test_noise_produces_spikes(self):
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0)
        params = translate_v7_noisy(bp)
        assert params["neuron_kwargs"]["noise_std"] == 1.0
        n = StochasticLIFNeuron(**params["neuron_kwargs"])
        spikes = 0
        for _ in range(5000):
            n.v += 18.0  # near threshold
            spikes += n.step(0.0)
        assert spikes > 0, "Noisy LIF near threshold should fire stochastically"
