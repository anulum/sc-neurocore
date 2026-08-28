# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rulkov map determinism contracts

"""Focused suite: TestRulkovDeterminism from former test_model_rulkov_map.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.rulkov_map import RulkovMapNeuron


class TestRulkovDeterminism:
    def test_bit_exact(self) -> None:
        traces = []
        for _ in range(2):
            n = RulkovMapNeuron()
            trace = [(n.step(1.0), n.x, n.y) for _ in range(300)]
            traces.append(trace)
        assert traces[0] == traces[1]
