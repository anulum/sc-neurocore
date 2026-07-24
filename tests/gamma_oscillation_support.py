# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_gamma_oscillation.py

from __future__ import annotations

"""Tests for the conductance-based PING circuit.

The behaviour-property tests (smoke spikes, no-drive silence,
inhibition-suppresses-firing, deterministic seeding) carry over
from the previous rate-coded implementation but now talk to the
new conductance-based API. The fidelity tests pin the published
30-80 Hz gamma peak from Börgers-Kopell 2003 Fig 2A and the
weak-PING gain-loop direction (raising w_ie suppresses E firing).
"""
import ctypes
import importlib
import os
import sys
from types import SimpleNamespace
from typing import Any, cast
import numpy as np
import pytest
from sc_neurocore.network import gamma_oscillation as gamma_oscillation_module
from tests.module_reload import restore_module_namespace, snapshot_module_namespace
from sc_neurocore.network.gamma_oscillation import (
    _HAS_GO_PING_STEP,
    _HAS_JULIA_PING_STEP,
    _HAS_MOJO_PING_STEP,
    _HAS_RUST_PING_STEP,
    PINGCircuit,
)

__all__ = [
    "ctypes",
    "importlib",
    "os",
    "sys",
    "SimpleNamespace",
    "Any",
    "cast",
    "np",
    "pytest",
    "gamma_oscillation_module",
    "restore_module_namespace",
    "snapshot_module_namespace",
    "_HAS_GO_PING_STEP",
    "_HAS_JULIA_PING_STEP",
    "_HAS_MOJO_PING_STEP",
    "_HAS_RUST_PING_STEP",
    "PINGCircuit",
]
