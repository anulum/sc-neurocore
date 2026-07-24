# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPatternDetection from former test_spike_train_stats_extended.py

"""Focused suite: TestPatternDetection from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from tests.spike_train_stats_extended_support import *  # noqa: F403


class TestPatternDetection:
    def test_unitary_events(self, population):
        events = unitary_events(population[:3])
        assert isinstance(events, list)

    def test_cell_assembly_detection(self, population):
        assemblies = cell_assembly_detection(population)
        assert isinstance(assemblies, list)

    def test_synfire_chain_detection(self, population):
        chains = synfire_chain_detection(population)
        assert isinstance(chains, list)
