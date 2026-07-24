# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExplanationResult from former test_explain.py

"""Focused suite: TestExplanationResult from former test_explain.py."""

from __future__ import annotations

from tests.explain_support import *  # noqa: F403


class TestExplanationResult:
    def test_top_k(self):
        imp = np.zeros((10, 5))
        imp[3, 2] = 1.0
        imp[7, 4] = 0.5
        r = ExplanationResult(method="test", importance_map=imp)
        top = r.top_k(2)
        assert len(top) == 2
        assert top[0] == (3, 2, 1.0)
        assert top[1] == (7, 4, 0.5)

    def test_summary(self):
        imp = np.random.rand(10, 5)
        r = ExplanationResult(method="test", importance_map=imp)
        s = r.summary()
        assert "test" in s
        assert "importance" in s
