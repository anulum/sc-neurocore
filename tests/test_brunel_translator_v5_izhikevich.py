# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestV5Izhikevich from former test_brunel_translator.py

"""Focused suite: TestV5Izhikevich from former test_brunel_translator.py."""

from __future__ import annotations

from tests.brunel_translator_support import *  # noqa: F403


class TestV5Izhikevich:
    """V5: Izhikevich regular-spiking neuron."""

    def test_fires_with_sustained_input(self):
        bp = BrunelParams(weight_exc=5.0, external_rate_hz=200.0)
        params = translate_v5_izhikevich(bp)
        n = SCIzhikevichNeuron(**params["neuron_kwargs"])
        spikes = sum(n.step(15.0) for _ in range(1000))
        assert spikes > 0, "Izhikevich must fire with sustained current"

    def test_threshold_is_30mv(self):
        bp = BrunelParams()
        params = translate_v5_izhikevich(bp)
        assert params["neuron_kwargs"]["c"] == -65.0
        assert params["neuron_kwargs"]["d"] == 8.0
