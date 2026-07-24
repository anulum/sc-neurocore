# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_qc_e2e.py

from __future__ import annotations

"""End-to-end tests for the quantum cognition pipeline.

Each scenario exercises multiple modules
in realistic conditions with production-scale parameters.
"""
import json
import math
import os
import subprocess
from pathlib import Path
import numpy as np
import pytest
from sc_neurocore.quantum_cognition.spin_pool import SpinPoolMPS
from sc_neurocore.quantum_cognition.fisher_posner import HybridFisherPosnerLIF
from sc_neurocore.quantum_cognition.bridge_adapter import (
    FisherPosnerQuantumBridge,
    compute_max_qubits,
    _get_available_ram,
)
from sc_neurocore.quantum_cognition.gotm_brain import GOTMBrain
from sc_neurocore.quantum_cognition.content_indexer import (
    ContentChunk,
    index_gotm_repo,
)
from sc_neurocore.quantum_cognition.radical_pair import (
    RadicalPairModel,
    RadicalPairParams,
)
from sc_neurocore.quantum_cognition.kane_mapper import (
    KaneSiliconMapper,
)
from sc_neurocore.quantum_cognition.studio_hook import QuantumStudioHook
from sc_neurocore.quantum_cognition.dashboard import TerminalDashboard

_QC_DIR = Path(__file__).resolve().parent.parent / "src" / "sc_neurocore" / "quantum_cognition"

__all__ = [
    "json",
    "math",
    "os",
    "subprocess",
    "Path",
    "np",
    "pytest",
    "SpinPoolMPS",
    "HybridFisherPosnerLIF",
    "FisherPosnerQuantumBridge",
    "compute_max_qubits",
    "_get_available_ram",
    "GOTMBrain",
    "ContentChunk",
    "index_gotm_repo",
    "RadicalPairModel",
    "RadicalPairParams",
    "KaneSiliconMapper",
    "QuantumStudioHook",
    "TerminalDashboard",
    "_QC_DIR",
    "__all__",
]
