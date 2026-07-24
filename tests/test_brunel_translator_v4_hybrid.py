# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestV4Hybrid from former test_brunel_translator.py

"""Focused suite: TestV4Hybrid from former test_brunel_translator.py."""

from __future__ import annotations

from tests.brunel_translator_support import *  # noqa: F403


class TestV4Hybrid:
    """V4: BitstreamSynapse AND + StochasticLIFNeuron."""

    def test_spike_from_high_prob_bitstream(self):
        """Sustained high-probability input bitstream should produce spikes."""
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0, weight_exc=5.0)
        params = translate_v4_hybrid(bp)
        n = StochasticLIFNeuron(**params["neuron_kwargs"])
        syn = BitstreamSynapse(**params["synapse_kwargs"])

        rng = np.random.default_rng(42)
        spikes = 0
        for _ in range(500):
            # High-probability pre-synaptic bitstream
            pre_bits = (rng.random(params["bitstream_length"]) < 0.9).astype(np.uint8)
            post_bits = syn.apply(pre_bits)
            current = post_bits.sum() * params["popcount_scale"]
            # Apply as delta-PSC voltage kick
            n.v += current
            spike = n.step(0.0)
            spikes += spike
        assert spikes > 0, "Hybrid SC+LIF must fire with high-probability drive"
