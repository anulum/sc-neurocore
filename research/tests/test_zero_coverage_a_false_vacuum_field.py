# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFalseVacuumField from former test_zero_coverage_a.py

"""Focused suite: TestFalseVacuumField from former test_zero_coverage_a.py."""

from __future__ import annotations

from typing import Any, cast


import numpy as np


class TestFalseVacuumField:
    """The false-vacuum field research module."""

    def test_nucleate_and_step(self) -> None:
        """After nucleation a step leaves a numeric energy (the check is on the type only)."""
        from transcendent.vacuum_decay import FalseVacuumField

        v = FalseVacuumField(size=8)
        v.nucleate(2, 2)
        cast(Any, v).step()
        e = v.measure_energy()
        assert isinstance(e, (int, float, np.integer, np.floating))
