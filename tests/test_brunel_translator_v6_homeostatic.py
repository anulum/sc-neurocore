# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestV6Homeostatic from former test_brunel_translator.py

"""Focused suite: TestV6Homeostatic from former test_brunel_translator.py."""

from __future__ import annotations

from tests.brunel_translator_support import *  # noqa: F403


class TestV6Homeostatic:
    """V6: HomeostaticLIFNeuron threshold adaptation."""

    def test_threshold_adapts(self):
        bp = BrunelParams(v_threshold=20.0, v_reset=10.0, weight_exc=5.0)
        params = translate_v6_homeostatic(bp)
        n = HomeostaticLIFNeuron(**params["neuron_kwargs"])
        initial_threshold = n.v_threshold
        for _ in range(500):
            n.v += 25.0
            n.step(0.0)
        assert n.v_threshold != initial_threshold, "Threshold must adapt"
