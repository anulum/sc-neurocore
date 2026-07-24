# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_predictive_coding.py

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.neuro_symbolic.predictive_coding import (
    HYPERVECTOR_DIM,
    Hypervector,
    PredictiveCodingLayer,
    ReasoningTrace,
    SymbolEncoder,
    VerifiableInference,
    _pack,
    _unpack,
)

__all__ = [
    "np",
    "pytest",
    "HYPERVECTOR_DIM",
    "Hypervector",
    "PredictiveCodingLayer",
    "ReasoningTrace",
    "SymbolEncoder",
    "VerifiableInference",
    "_pack",
    "_unpack",
]
