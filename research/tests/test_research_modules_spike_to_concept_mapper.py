# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeToConceptMapper from former test_research_modules.py

"""Focused suite: TestSpikeToConceptMapper from former test_research_modules.py."""

from __future__ import annotations

from sc_neurocore.analysis.explainability import SpikeToConceptMapper
import numpy as np


class TestSpikeToConceptMapper:
    """Explaining spikes as named concepts."""

    def test_active_concepts(self) -> None:
        """Active vision and audio neurons are named."""
        mapper = SpikeToConceptMapper({0: "Vision", 1: "Motor", 2: "Audio"})
        spikes = np.array([1, 0, 1, 0])
        result = mapper.explain(spikes)
        assert "Vision" in result
        assert "Audio" in result

    def test_idle(self) -> None:
        """No spikes are explained as idle."""
        mapper = SpikeToConceptMapper({0: "Vision"})
        spikes = np.array([0, 0, 0])
        assert "idle" in mapper.explain(spikes)

    def test_unknown_neuron(self) -> None:
        """An unmapped neuron is named as unknown with its index."""
        mapper = SpikeToConceptMapper({0: "Vision"})
        spikes = np.array([0, 1])  # neuron 1 not in map
        result = mapper.explain(spikes)
        assert "Unknown(1)" in result
