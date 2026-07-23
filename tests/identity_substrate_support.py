# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_identity_substrate.py

from __future__ import annotations

"""Tests for the identity continuity substrate (Phase 1)."""
import numpy as np
from sc_neurocore.identity import (
    IdentitySubstrate,
    TraceEncoder,
    StateDecoder,
    Checkpoint,
    DirectorController,
)
N_CORTICAL = 20
N_INHIBITORY = 8
N_MEMORY = 5
def _make_substrate(seed=42):
    return IdentitySubstrate(N_CORTICAL, N_INHIBITORY, N_MEMORY, seed=seed)

__all__ = ['np', 'IdentitySubstrate', 'TraceEncoder', 'StateDecoder', 'Checkpoint', 'DirectorController', 'N_CORTICAL', 'N_INHIBITORY', 'N_MEMORY', '_make_substrate']
