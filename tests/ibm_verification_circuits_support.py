# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_ibm_verification_circuits.py

"""Shared Posner verification-circuit imports and helpers, behind the tools path and the qiskit and scipy guards."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
qiskit = pytest.importorskip("qiskit")
pytest.importorskip("scipy")

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

import verify_ibm_heron as vih
from verify_ibm_heron import (
    REFERENCE_TEST_HF_SITE1,
    REFERENCE_TEST_HF_SITE2,
    DEFAULT_NUC_DIPOLAR,
    DEFAULT_NUC_DIPOLAR_CROSS,
    _INTRA_PAIRS,
    _CROSS_PAIRS,
    _DIPOLAR_PAIRS,
    analyse_chain,
    analyse_rpm_8q,
    analytical_singlet_thermal,
    analytical_singlet_recombination,
    analytical_chain_corr,
    _posner_chain_couplings,
    _parse_dipolar_pairs,
    build_posner_circuit,
    build_posner_decoherence_circuit,
    build_chain_circuit,
    posner_hamiltonian,
)
from qiskit.quantum_info import Statevector

HF1 = REFERENCE_TEST_HF_SITE1
HF2 = REFERENCE_TEST_HF_SITE2
ZERO_TENSOR = {"Axx": 0.0, "Ayy": 0.0, "Azz": 0.0, "Axy": 0.0, "Axz": 0.0, "Ayz": 0.0}
INCORPORATION_TEST_TENSORS = {
    "p1_p2": {"Axx": 0.10, "Ayy": 0.10, "Azz": 0.05, "Axy": 0.01, "Axz": 0.0, "Ayz": 0.0},
    "p1_p3": {"Axx": 0.04, "Ayy": 0.04, "Azz": 0.02, "Axy": 0.0, "Axz": 0.01, "Ayz": 0.0},
    "p2_p3": {"Axx": 0.02, "Ayy": 0.02, "Azz": 0.03, "Axy": 0.0, "Axz": 0.0, "Ayz": 0.01},
}
CA_ELECTRON_MAP_TEST = {
    8: 0,
    11: 0,
    14: 0,
    17: 1,
    20: 1,
    23: 1,
    26: 0,
    29: 1,
    32: 0,
}
CA43_ZERO_TENSORS = {start: ZERO_TENSOR for start in CA_ELECTRON_MAP_TEST}


def _sv(qc: QuantumCircuit, shots: int = 100_000) -> dict[str, int]:
    """Sample measurement counts from the ideal statevector of ``qc``."""
    counts: dict[str, int] = Statevector.from_instruction(
        qc.remove_final_measurements(inplace=False)
    ).sample_counts(shots)
    return counts


def _angle_sum(qc: QuantumCircuit, gate_name: str, qa: int, qb: int) -> float:
    """Sum the angles of ``gate_name`` gates acting on exactly qubits ``qa`` and ``qb``."""
    total = 0.0
    target = {qa, qb}
    for circuit_instruction in qc.data:
        inst = circuit_instruction.operation
        if inst.name != gate_name:
            continue
        qubits = {qc.find_bit(qarg).index for qarg in circuit_instruction.qubits}
        if qubits == target:
            total += float(inst.params[0])
    return total


__all__ = [
    "math",
    "sys",
    "Path",
    "np",
    "pytest",
    "qiskit",
    "vih",
    "REFERENCE_TEST_HF_SITE1",
    "REFERENCE_TEST_HF_SITE2",
    "DEFAULT_NUC_DIPOLAR",
    "DEFAULT_NUC_DIPOLAR_CROSS",
    "_INTRA_PAIRS",
    "_CROSS_PAIRS",
    "_DIPOLAR_PAIRS",
    "analyse_chain",
    "analyse_rpm_8q",
    "analytical_singlet_thermal",
    "analytical_singlet_recombination",
    "analytical_chain_corr",
    "_posner_chain_couplings",
    "_parse_dipolar_pairs",
    "build_posner_circuit",
    "build_posner_decoherence_circuit",
    "build_chain_circuit",
    "posner_hamiltonian",
    "Statevector",
    "HF1",
    "HF2",
    "ZERO_TENSOR",
    "INCORPORATION_TEST_TENSORS",
    "CA_ELECTRON_MAP_TEST",
    "CA43_ZERO_TENSORS",
    "_sv",
    "_angle_sum",
    "__all__",
]
