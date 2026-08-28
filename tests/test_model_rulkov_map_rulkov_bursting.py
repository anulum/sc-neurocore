# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rulkov map bursting contracts

"""Focused suite: TestRulkovBursting from former test_model_rulkov_map.py."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.rulkov_map import RulkovMapNeuron
from tests.model_rulkov_map_support import _run


class TestRulkovBursting:
    """Burst detection: short ISIs within burst, long ISIs between bursts."""

    def test_short_isi_within_burst(self) -> None:
        """At I=0.5, spikes come in rapid clusters (ISI ~5-6 steps)."""
        n = RulkovMapNeuron()
        spikes = _run(n, current=0.5, steps=50000)
        assert len(spikes) >= 10
        isis = np.diff(spikes)
        # Most ISIs should be short (within-burst)
        median_isi = np.median(isis)
        assert median_isi < 10, f"Median ISI = {median_isi}, expected short bursts"

    def test_isi_variability(self) -> None:
        """ISIs should show variability (mix of intra- and inter-burst intervals)."""
        n = RulkovMapNeuron()
        spikes = _run(n, current=0.5, steps=50000)
        if len(spikes) >= 10:
            isis = np.diff(spikes).astype(float)
            cv = np.std(isis) / np.mean(isis)
            # Map dynamics produce variable ISIs (not perfectly regular)
            assert cv > 0.1, f"CV(ISI) = {cv:.4f}, expected variability"
