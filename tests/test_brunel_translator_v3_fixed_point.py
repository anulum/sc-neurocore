# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestV3FixedPoint from former test_brunel_translator.py

"""Focused suite: TestV3FixedPoint from former test_brunel_translator.py."""

from __future__ import annotations

from tests.brunel_translator_support import *  # noqa: F403


class TestV3FixedPoint:
    """V3: FixedPointLIFNeuron Q8.8."""

    def test_no_overflow_brunel_weights(self):
        """Max Brunel weight must not overflow Q8.8 signed range."""
        bp = BrunelParams(weight_exc=5.0, g_inh=5.0)
        params = translate_v3_fixed_point(bp)
        # Q8.8 signed range: -32768 to 32767
        assert -32768 <= params["j_exc_q"] <= 32767
        assert -32768 <= params["j_inh_q"] <= 32767
        assert -32768 <= params["v_threshold_q"] <= 32767

    def test_single_neuron_fires(self):
        """Q8.8 neuron should fire with sustained suprathreshold input."""
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0, weight_exc=5.0)
        params = translate_v3_fixed_point(bp)
        n = FixedPointLIFNeuron(
            data_width=params["data_width"],
            fraction=params["fraction"],
            v_threshold=params["v_threshold_q"],
            v_reset=params["v_reset_q"],
            refractory_period=params["refractory_period"],
        )
        spikes = 0
        for _ in range(1000):
            # Drive with weight as current input
            spike, _ = n.step(
                leak_k=params["leak_k"],
                gain_k=params["gain_k"],
                I_t=params["j_exc_q"] * 10,  # 10 simultaneous inputs
            )
            spikes += spike
        assert spikes > 0, "Q8.8 neuron must fire with sustained input"
