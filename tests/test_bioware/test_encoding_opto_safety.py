# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOptoSafety from former test_encoding.py

"""Focused suite: TestOptoSafety from former test_encoding.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.bioware.bioware import (
    SCToOptoEncoder,
)


class TestOptoSafety:
    """Safety caps on optogenetic stimulation."""

    def test_power_cap(self) -> None:
        """Total optical power stays within its cap."""
        # Create many active neurons exceeding total cap
        bs = {i: np.ones(100, dtype=np.uint8) for i in range(100)}
        enc = SCToOptoEncoder(
            max_intensity_mw_mm2=5.0,
            max_total_power_mw=10.0,
        )
        pulses = enc.encode(bs)
        total_mw = sum(p.power_mw for p in pulses)
        assert total_mw <= 10.0

    def test_no_cap_violation_with_few(self) -> None:
        """A single active neuron gets its pulse."""
        bs = {0: np.ones(100, dtype=np.uint8)}
        enc = SCToOptoEncoder(max_total_power_mw=50.0)
        pulses = enc.encode(bs)
        assert len(pulses) == 1
