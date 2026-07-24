# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_quantum_cognition.py

from __future__ import annotations

"""Comprehensive tests for the quantum cognition (Fisher-Posner) layer.

Covers: SpinPoolMPS, HybridFisherPosnerLIF, FisherPosnerQuantumBridge,
QuantumStudioHook, and cross-module non-locality verification.
"""
import json
import numpy as np
import pytest
from sc_neurocore.quantum_cognition.spin_pool import SpinCouplingTensor, SpinPoolMPS
from sc_neurocore.quantum_cognition.fisher_posner import HybridFisherPosnerLIF
from sc_neurocore.quantum_cognition.bridge_adapter import (
    FisherPosnerQuantumBridge,
    HAS_PENNYLANE,
)
from sc_neurocore.quantum_cognition.studio_hook import (
    QuantumStudioHook,
    QuantumCognitionLayerMetadata,
)

__all__ = [
    "json",
    "np",
    "pytest",
    "SpinCouplingTensor",
    "SpinPoolMPS",
    "HybridFisherPosnerLIF",
    "FisherPosnerQuantumBridge",
    "HAS_PENNYLANE",
    "QuantumStudioHook",
    "QuantumCognitionLayerMetadata",
]
