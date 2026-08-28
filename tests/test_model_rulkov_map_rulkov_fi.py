# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rulkov map current-response contracts

"""Focused suite: TestRulkovFI from former test_model_rulkov_map.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.rulkov_map import RulkovMapNeuron
from tests.model_rulkov_map_support import _run


class TestRulkovFI:
    def test_no_spikes_at_zero_input(self) -> None:
        """Default params (sigma=-1.6) → no spikes at I=0."""
        n = RulkovMapNeuron()
        spikes = len(_run(n, current=0.0, steps=50000))
        assert spikes == 0

    def test_current_triggers_spiking(self) -> None:
        """I=0.5 drives the map above threshold."""
        n = RulkovMapNeuron()
        spikes = len(_run(n, current=0.5, steps=50000))
        assert spikes > 10

    def test_rate_increases_with_current(self) -> None:
        n1 = RulkovMapNeuron()
        n5 = RulkovMapNeuron()
        s1 = len(_run(n1, current=0.5, steps=50000))
        s5 = len(_run(n5, current=5.0, steps=50000))
        assert s5 > s1

    def test_monotonic_fi(self) -> None:
        rates = []
        for I in [0.5, 1.0, 2.0, 5.0]:
            n = RulkovMapNeuron()
            rates.append(len(_run(n, current=I, steps=50000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
