# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGenomeDiff from former test_statistics.py

"""Focused suite: TestGenomeDiff from former test_statistics.py."""

from __future__ import annotations

from tests.test_evo_substrate.statistics_support import *  # noqa: F403

class TestGenomeDiff:
    def test_identical(self) -> None:
        g = Genome()
        d = genome_diff(g, g)
        assert d.is_identical
        assert d.neuron_delta == 0

    def test_different(self) -> None:
        a = Genome()
        b = Genome()
        b.topology.num_neurons = 64
        d = genome_diff(a, b)
        assert not d.is_identical
        assert d.neuron_delta == 48
