# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNestedUniverse from former test_zero_coverage_a.py

"""Focused suite: TestNestedUniverse from former test_zero_coverage_a.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure same-dir support module is importable under pytest importlib mode.
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from zero_coverage_a_support import *  # noqa: F403

class TestNestedUniverse:
    def test_spawn_and_step(self):
        from eschaton.simulation import NestedUniverse

        u = NestedUniverse(id=0, computing_resources=100.0)
        child = u.spawn_simulation(overhead=0.1)
        assert child is not None
        assert child in u.children
        u.run_recursive_step()
