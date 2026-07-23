# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_sc_quantum_compiler.py

"""Module-level tests from former test_sc_quantum_compiler.py."""

from __future__ import annotations

from tests.sc_quantum_compiler_support import *  # noqa: F403

def test_quantum_package_exports_sc_compiler_surface() -> None:
    """The quantum package facade exposes the documented SC compiler surface."""
    expected_names = {
        "QuantumGate",
        "SCQuantumCircuit",
        "compile_sc_layer",
        "compile_sc_multiply",
        "prob_to_ry_angle",
        "ry_gate",
        "sc_prob_to_statevector",
        "statevector_to_prob",
    }

    assert expected_names <= set(quantum.__all__)
    assert quantum.compile_sc_multiply is compile_sc_multiply
    assert quantum.compile_sc_layer is compile_sc_layer
