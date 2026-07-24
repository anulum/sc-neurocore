# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNoveltyArchive from former test_ecology.py

"""Focused suite: TestNoveltyArchive from former test_ecology.py."""

from __future__ import annotations

from tests.test_evo_substrate.ecology_support import *  # noqa: F403


class TestNoveltyArchive:
    def test_empty_archive_high_score(self) -> None:
        na = NoveltyArchive()
        assert na.novelty_score(np.array([1.0, 2.0])) == 1.0

    def test_add_novel(self) -> None:
        na = NoveltyArchive(threshold=0.01)
        assert na.maybe_add(np.array([1.0, 0.0]))
        assert na.size == 1

    def test_add_duplicate_rejected(self) -> None:
        na = NoveltyArchive(threshold=0.5)
        na.maybe_add(np.array([1.0, 0.0]))
        assert not na.maybe_add(np.array([1.0, 0.0]))  # identical
