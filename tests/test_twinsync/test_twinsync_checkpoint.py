# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCheckpoint from former test_twinsync.py

"""Focused suite: TestCheckpoint from former test_twinsync.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.digital_twin.twinsync import (
    Checkpoint,
)


class TestCheckpoint:
    """Checksums of a node checkpoint."""

    def test_checksum(self) -> None:
        """The checksum is 16 characters long."""
        cp = Checkpoint(0, 1000, 0, neuron_state=np.array([1.0, 2.0]))
        cs = cp.compute_checksum()
        assert len(cs) == 16

    def test_checksum_deterministic(self) -> None:
        """The same checkpoint gives the same checksum twice."""
        cp = Checkpoint(0, 1000, 0, lfsr_state=42)
        assert cp.compute_checksum() == cp.compute_checksum()
