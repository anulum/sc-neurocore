# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_cortical_column.py

from __future__ import annotations

"""Tests for the 8-population cortical microcircuit.

Smoke and determinism tests use `scale=0.02` with
`scale_correction=False` so that they finish in under ~5 s. Fidelity
tests against Potjans Table 4 require the published lower-bound
`scale=0.1` with full-scale in-degree preservation; those tests run
~25 s and are isolated to the `TestPublishedFidelity` class so they
can be filtered with `pytest -k 'not Fidelity'` for fast iteration.
"""
import importlib
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from sc_neurocore.network import cortical_column as cortical_column_module
from tests.module_reload import restore_module_namespace, snapshot_module_namespace
from sc_neurocore.network.cortical_column import (
    CONN_PROBS,
    CorticalColumn,
    FULL_SIZES,
    K_BG,
    POPULATIONS,
)
@pytest.fixture(scope="class")
def rasters():
    col = CorticalColumn(scale=0.1, scale_correction=True, seed=42)
    return col, col.simulate(duration_ms=600.0, dt=0.1)

__all__ = ['importlib', 'sys', 'SimpleNamespace', 'np', 'pytest', 'sparse', 'cortical_column_module', 'restore_module_namespace', 'snapshot_module_namespace', 'CONN_PROBS', 'CorticalColumn', 'FULL_SIZES', 'K_BG', 'POPULATIONS', 'rasters']
