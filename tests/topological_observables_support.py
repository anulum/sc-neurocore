# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_topological_observables.py

from __future__ import annotations

"""Tests for winding_number, ollivier_ricci_curvature,
sheaf_consistency_defect, connection_curvature."""
import numpy as np
from sc_neurocore.math.topology import (
    winding_number,
    ollivier_ricci_curvature,
    sheaf_consistency_defect,
    connection_curvature,
)

__all__ = ['np', 'winding_number', 'ollivier_ricci_curvature', 'sheaf_consistency_defect', 'connection_curvature']
