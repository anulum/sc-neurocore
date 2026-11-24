# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSingleton from former test_tcbo_demo.py

"""Focused suite: TestSingleton from former test_tcbo_demo.py."""

from __future__ import annotations

import unittest
from sc_neurocore.experiments.tcbo_demo_engine import (
    get_tcbo_demo_engine,
    reset_tcbo_demo_engine,
)


class TestSingleton(unittest.TestCase):
    """The shared TCBO demo engine."""

    def test_singleton(self) -> None:
        """Two lookups return the same engine."""
        reset_tcbo_demo_engine()
        e1 = get_tcbo_demo_engine()
        e2 = get_tcbo_demo_engine()
        self.assertIs(e1, e2)

    def test_reset(self) -> None:
        """A reset makes the next lookup return a new engine."""
        e1 = get_tcbo_demo_engine()
        reset_tcbo_demo_engine()
        e2 = get_tcbo_demo_engine()
        self.assertIsNot(e1, e2)
