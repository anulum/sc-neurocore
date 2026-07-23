# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_scpn_integrated.py

from __future__ import annotations

"""Tests for create_full_stack, run_integrated_step, get_global_metrics, K_nm matrix."""
import numpy as np
import pytest
from sc_neurocore.scpn import (
    create_full_stack,
    run_integrated_step,
    get_global_metrics,
)
from sc_neurocore.scpn.layers import LAYER_REGISTRY
from sc_neurocore.scpn.params import build_knm_matrix, OMEGA_N

__all__ = ['np', 'pytest', 'create_full_stack', 'run_integrated_step', 'get_global_metrics', 'LAYER_REGISTRY', 'build_knm_matrix', 'OMEGA_N']
