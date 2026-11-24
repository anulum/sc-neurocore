# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCatomLattice from former test_zero_coverage_a.py

"""Focused suite: TestCatomLattice from former test_zero_coverage_a.py."""

from __future__ import annotations

from typing import Any, cast


import numpy as np


class TestCatomLattice:
    """The claytronic catom lattice research module."""

    def test_reconfigure_and_topology(self) -> None:
        """After reconfiguration the topology is an array (the check is on the type only)."""
        from post_silicon.claytronics import CatomLattice

        c = CatomLattice(size=4)
        cast(Any, c).reconfigure()
        topo = c.get_topology()
        assert isinstance(topo, np.ndarray)
