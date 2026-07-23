# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestV2RateMatched from former test_brunel_translator.py

"""Focused suite: TestV2RateMatched from former test_brunel_translator.py."""

from __future__ import annotations

from tests.brunel_translator_support import *  # noqa: F403

class TestV2RateMatched:
    """V2: VectorizedSCLayer probability domain."""

    def test_output_proportional_to_input(self):
        """Mean output probability should increase with input probability."""
        bp = BrunelParams()
        params = translate_v2_rate_matched(bp)
        layer = VectorizedSCLayer(n_inputs=4, n_neurons=2, length=params["bitstream_length"])
        # High input probability
        out_high = layer.forward([0.8, 0.8, 0.8, 0.8])
        # Low input probability
        out_low = layer.forward([0.1, 0.1, 0.1, 0.1])
        assert out_high.mean() > out_low.mean()
