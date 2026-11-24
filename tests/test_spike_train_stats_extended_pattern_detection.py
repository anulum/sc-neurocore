# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPatternDetection from former test_spike_train_stats_extended.py

"""Focused suite: TestPatternDetection from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from typing import Any

import numpy as np

from sc_neurocore.analysis.spike_stats import cell_assembly_detection
from sc_neurocore.analysis.spike_stats import synfire_chain_detection
from sc_neurocore.analysis.spike_stats import unitary_events
from tests.spike_train_stats_extended_support import population as population


class TestPatternDetection:
    """Unitary events, cell assemblies and synfire chains on a five-train population."""

    def test_unitary_events(self, population: list[np.ndarray[Any, Any]]) -> None:
        """Unitary-event analysis of three trains returns a list (the check is on the type only)."""
        events = unitary_events(population[:3])
        assert isinstance(events, list)

    def test_cell_assembly_detection(self, population: list[np.ndarray[Any, Any]]) -> None:
        """Cell-assembly detection returns a list (the check is on the type only)."""
        assemblies = cell_assembly_detection(population)
        assert isinstance(assemblies, list)

    def test_synfire_chain_detection(self, population: list[np.ndarray[Any, Any]]) -> None:
        """Synfire-chain detection returns a list (the check is on the type only)."""
        chains = synfire_chain_detection(population)
        assert isinstance(chains, list)
