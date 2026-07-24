# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAutoTargetRecommender from former test_platforms.py

"""Focused suite: TestAutoTargetRecommender from former test_platforms.py."""

from __future__ import annotations

from tests.platforms_support import *  # noqa: F403


class TestAutoTargetRecommender:
    def test_basic_recommendation(self):
        from sc_neurocore.compiler.intelligence import recommend_target

        recs = recommend_target({"v": "a + b"})
        assert len(recs) == 5
        assert recs[0].score >= recs[-1].score

    def test_class_filter(self):
        from sc_neurocore.compiler.intelligence import recommend_target

        recs = recommend_target(
            {"v": "a * b + c"},
            require_class="neuromorphic",
        )
        for r in recs:
            assert "neuromorphic" in r.rationale

    def test_width_filter(self):
        from sc_neurocore.compiler.intelligence import recommend_target

        recs = recommend_target({"v": "a + b"}, max_data_width=8)
        from sc_neurocore.compiler.platforms import get_profile

        for r in recs:
            assert get_profile(r.profile_name).data_width <= 8

    def test_top_n(self):
        from sc_neurocore.compiler.intelligence import recommend_target

        recs = recommend_target({"v": "a + b"}, top_n=3)
        assert len(recs) == 3
