# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDendrifyDynamics from former test_model_dendrify.py

"""Focused suite: TestDendrifyDynamics from former test_model_dendrify.py."""

from __future__ import annotations

from tests.model_dendrify_support import *  # noqa: F403


class TestDendrifyDynamics:
    def test_subthreshold_silent(self):
        n = DendrifyNeuron()
        assert len(_run(n, current=10.0, steps=10000)) == 0

    def test_suprathreshold_fires(self):
        n = DendrifyNeuron()
        assert len(_run(n, current=50.0, steps=10000)) >= 50

    def test_rate_increases(self):
        n50 = DendrifyNeuron()
        n100 = DendrifyNeuron()
        s50 = len(_run(n50, current=50.0, steps=10000))
        s100 = len(_run(n100, current=100.0, steps=10000))
        assert s100 > s50

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = DendrifyNeuron()
            trace = [(n.step(50.0), n.v_s) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
