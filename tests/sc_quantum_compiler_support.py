# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_sc_quantum_compiler.py

from __future__ import annotations

"""Tests for SC-to-quantum compilation (Conjecture C1+C4)."""
import numpy as np
import sc_neurocore.quantum as quantum
from sc_neurocore.quantum.sc_quantum_compiler import (
    sc_prob_to_statevector,
    statevector_to_prob,
    prob_to_ry_angle,
    ry_gate,
    compile_sc_multiply,
    compile_sc_layer,
)

__all__ = [
    "np",
    "quantum",
    "sc_prob_to_statevector",
    "statevector_to_prob",
    "prob_to_ry_angle",
    "ry_gate",
    "compile_sc_multiply",
    "compile_sc_layer",
]
