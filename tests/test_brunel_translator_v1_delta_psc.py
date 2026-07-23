# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestV1DeltaPSC from former test_brunel_translator.py

"""Focused suite: TestV1DeltaPSC from former test_brunel_translator.py."""

from __future__ import annotations

from tests.brunel_translator_support import *  # noqa: F403

class TestV1DeltaPSC:
    """V1: StochasticLIF with delta-PSC wiring."""

    def test_suprathreshold_epsp_fires(self):
        """Single voltage kick >= v_threshold must produce a spike."""
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0)
        params = translate_v1_stochastic_lif(bp)
        n = StochasticLIFNeuron(**params["neuron_kwargs"])
        # Delta-PSC: direct voltage jump above threshold
        n.v += 21.0
        spike = n.step(0.0)  # leak-only step
        assert spike == 1
        assert n.v == bp.v_reset

    def test_subthreshold_no_spike(self):
        """Input below threshold must not fire."""
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0)
        params = translate_v1_stochastic_lif(bp)
        n = StochasticLIFNeuron(**params["neuron_kwargs"])
        n.v += 5.0
        spike = n.step(0.0)
        assert spike == 0
