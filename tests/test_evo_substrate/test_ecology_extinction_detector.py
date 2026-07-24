# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExtinctionDetector from former test_ecology.py

"""Focused suite: TestExtinctionDetector from former test_ecology.py."""

from __future__ import annotations

from tests.test_evo_substrate.ecology_support import *  # noqa: F403


class TestExtinctionDetector:
    def test_no_extinction_early(self) -> None:
        ed = ExtinctionDetector(stagnation_gens=5)
        for i in range(3):
            assert ed.check(0.5) is False

    def test_detects_stagnation(self) -> None:
        ed = ExtinctionDetector(stagnation_gens=5)
        for _ in range(10):
            ed.check(0.5)  # all same fitness
        assert ed.extinction_count > 0

    def test_apply_kills(self) -> None:
        ed = ExtinctionDetector(kill_fraction=0.5)
        pop = [Organism(genome=Genome()) for _ in range(10)]
        rng = np.random.default_rng(42)
        killed = ed.apply(pop, rng)
        assert killed == 5

    def test_improving_history_does_not_trigger_extinction(self) -> None:
        ed = ExtinctionDetector(stagnation_gens=3)

        assert ed.check(0.1) is False
        assert ed.check(0.2) is False
        assert ed.check(0.3) is False
        assert ed.extinction_count == 0
