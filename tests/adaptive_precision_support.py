# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_adaptive_precision.py

from __future__ import annotations

"""Tests for the adaptive precision assignment module."""
import json
from pathlib import Path
from collections.abc import Callable
from typing import cast
import numpy as np
import pytest
from sc_neurocore.compiler import formal_property_check
from sc_neurocore.compiler.adaptive_precision import (
    auto_tune_synapse_precisions,
    LayerPrecision,
    SynapsePrecision,
    analyze_sensitivity,
    assign_lengths,
    assign_synapse_precisions,
    precision_plan_manifest,
    write_precision_formal_evidence_bundle,
)
from sc_neurocore.compiler.formal_property_check import PropertyProofResult
_needs_formal = pytest.mark.skipif(
    not formal_property_check.formal_tools_available(),
    reason="SymbiYosys / Yosys / solver not available",
)

__all__ = ['json', 'Path', 'Callable', 'cast', 'np', 'pytest', 'formal_property_check', 'auto_tune_synapse_precisions', 'LayerPrecision', 'SynapsePrecision', 'analyze_sensitivity', 'assign_lengths', 'assign_synapse_precisions', 'precision_plan_manifest', 'write_precision_formal_evidence_bundle', 'PropertyProofResult', '_needs_formal']
