# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeToConceptMapper from former test_bio_chaos_spatial_learning.py

"""Focused suite: TestSpikeToConceptMapper from former test_bio_chaos_spatial_learning.py."""

from __future__ import annotations

from tests.bio_chaos_spatial_learning_support import *  # noqa: F403


class TestSpikeToConceptMapper:
    def test_active_spikes(self):
        mapper = SpikeToConceptMapper({0: "Motor", 2: "Vision"})
        out = mapper.explain(np.array([1, 0, 1, 0]))
        assert "Motor" in out and "Vision" in out

    def test_no_spikes(self):
        assert "idle" in SpikeToConceptMapper({0: "Motor"}).explain(np.array([0, 0, 0]))

    def test_unknown_index(self):
        assert "Unknown(1)" in SpikeToConceptMapper({0: "Motor"}).explain(np.array([0, 1]))

    def test_empty_concept_map(self):
        out = SpikeToConceptMapper({}).explain(np.array([1, 1]))
        assert "Unknown(0)" in out and "Unknown(1)" in out
