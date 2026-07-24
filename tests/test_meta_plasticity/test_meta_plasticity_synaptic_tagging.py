# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSynapticTagging from former test_meta_plasticity.py

"""Focused suite: TestSynapticTagging from former test_meta_plasticity.py."""

from __future__ import annotations

from meta_plasticity_support import *  # noqa: F403


class TestSynapticTagging:
    def test_create_tag(self):
        tm = TaggingModel()
        tag = tm.create_tag(synapse_id=0, strength=0.8, time_ms=100.0)
        assert tag.tag_strength == 0.8
        assert not tag.captured

    def test_decay_reduces_strength(self):
        tm = TaggingModel(tag_decay_rate=0.1)
        tag = tm.create_tag(0, 0.8, 0.0)
        tm.decay_tags(10.0)
        assert tag.tag_strength < 0.8

    def test_consolidate_captures(self):
        tm = TaggingModel(capture_threshold=0.3)
        tm.create_tag(0, 0.5, 0.0)
        captured = tm.consolidate(consolidation_strength=0.8)
        assert captured == 1
        assert tm.tags[0].captured

    def test_consolidate_weak_signal(self):
        tm = TaggingModel(capture_threshold=0.3)
        tm.create_tag(0, 0.5, 0.0)
        captured = tm.consolidate(consolidation_strength=0.2)
        assert captured == 0

    def test_prune_expired(self):
        tm = TaggingModel()
        tm.create_tag(0, 0.001, 0.0)  # below 0.01 threshold
        pruned = tm.prune_expired()
        assert pruned == 1

    def test_active_tags(self):
        tm = TaggingModel()
        tm.create_tag(0, 0.5, 0.0)
        tm.create_tag(1, 0.005, 0.0)  # expired
        assert tm.active_tags == 1
