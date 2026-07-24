# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestShortTermPlasticity from former test_learning_advanced.py

"""Focused suite: TestShortTermPlasticity from former test_learning_advanced.py."""

from __future__ import annotations

from tests.learning_advanced_support import *  # noqa: F403


class TestShortTermPlasticity:
    def test_class_exists(self):
        """ShortTermPlasticity should be importable."""
        assert ShortTermPlasticity is not None
