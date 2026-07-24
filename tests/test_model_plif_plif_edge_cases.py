# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPLIFEdgeCases from former test_model_plif.py

"""Focused suite: TestPLIFEdgeCases from former test_model_plif.py."""

from __future__ import annotations

from tests.model_plif_support import *  # noqa: F403


class TestPLIFEdgeCases:
    def test_zero_input(self):
        """Zero input from rest → V stays at 0, no spikes."""
        n = ParametricLIFNeuron()
        spikes = sum(n.step(0.0) for _ in range(100))
        assert spikes == 0
        assert n.v == 0.0

    def test_negative_input(self):
        """Negative input drives V below 0."""
        n = ParametricLIFNeuron()
        n.step(-0.5)
        assert n.v == -0.5

    def test_reset_method(self):
        n = ParametricLIFNeuron()
        for _ in range(50):
            n.step(2.0)
        n.reset()
        assert n.v == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = ParametricLIFNeuron(a=1.0)
            trace = [(n.step(0.5), n.v) for _ in range(100)]
            traces.append(trace)
        assert traces[0] == traces[1]
