# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_identity_director.py

from __future__ import annotations

"""Exercise DirectorController.diagnose(), correct(), report(), and helpers."""
from unittest.mock import patch
import numpy as np
from sc_neurocore.identity.substrate import IdentitySubstrate
from sc_neurocore.identity.director import (
    DirectorController,
    _add_weight_noise,
    _homeostatic_scale,
    _prune_weak,
    _grow_synapses,
)


def _make_substrate():
    sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
    sub.run(duration=0.1, dt=0.001)
    return sub


__all__ = [
    "patch",
    "np",
    "IdentitySubstrate",
    "DirectorController",
    "_add_weight_noise",
    "_homeostatic_scale",
    "_prune_weak",
    "_grow_synapses",
    "_make_substrate",
]
