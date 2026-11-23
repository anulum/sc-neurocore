# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWindingNumber from former test_topology.py

"""Focused suite: TestWindingNumber from former test_topology.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.math.topology import (
    winding_number,
)


class TestWindingNumber:
    """The winding number of a phase sequence."""

    def test_one_full_rotation(self) -> None:
        """One rotation gives winding number one."""
        phases = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        assert winding_number(phases) == 1

    def test_two_rotations(self) -> None:
        """Two rotations give winding number two."""
        phases = np.linspace(0, 4 * np.pi, 200, endpoint=False)
        assert winding_number(phases) == 2

    def test_no_rotation(self) -> None:
        """No rotation gives winding number zero."""
        phases = np.ones(50) * 1.5
        assert winding_number(phases) == 0

    def test_negative_rotation(self) -> None:
        """A backwards rotation gives winding number minus one."""
        phases = np.linspace(2 * np.pi, 0, 100, endpoint=False)
        assert winding_number(phases) == -1
