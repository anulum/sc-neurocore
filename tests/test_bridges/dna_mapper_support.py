# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_dna_mapper.py

from __future__ import annotations

"""Multi-angle test suite for the DNA computing mapper.

Tests cover:
- Sequence design constraints (GC content, homopolymer, orthogonality)
- Gate compilation correctness (AND, OR, NOT, THRESHOLD, NAND, XOR)
- Circuit assembly and validation
- Kinetic simulation convergence
- Export format correctness (GenBank, FASTA, NUPACK, JSON)
- Edge cases (empty network, single gate, maximum complexity)
- Round-trip fidelity (compile → simulate → verify logic)
- Thermodynamic validation
- Strand interaction analysis
"""
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import numpy as np
import pytest
import sc_neurocore.bridges.dna_mapper as dna_mapper
from sc_neurocore.bridges.dna_mapper import (
    BitstreamToDNA,
    CompilationMethod,
    ConcentrationOptimizer,
    CrossHybridizationChecker,
    DNACircuitDesign,
    DNAGate,
    DNAStrand,
    DegradationModel,
    DualRailEncoder,
    EnzymaticGateCompiler,
    GF4ErrorCorrection,
    GateOptimizer,
    GateType,
    HairpinChecker,
    KineticSimulator,
    NUPACKInterface,
    NoiseModel,
    PlateLayout,
    SCNetworkBridge,
    SCPrecisionAnalyzer,
    SequenceDesigner,
    StrandDisplacementCompiler,
    TopologicalAnalyzer,
    estimate_cost,
    export_fasta,
    export_genbank,
    export_json,
    export_nupack_input,
    generate_protocol,
    visualize_circuit,
    visualize_kinetics,
)
from sc_neurocore.bridges.dna_types import _GC_TARGET_HIGH, _GC_TARGET_LOW, _MAX_HOMOPOLYMER
@pytest.fixture
def designer() -> SequenceDesigner:
    return SequenceDesigner(seed=42)
@pytest.fixture
def displacement_compiler() -> StrandDisplacementCompiler:
    return StrandDisplacementCompiler()
@pytest.fixture
def enzymatic_compiler() -> EnzymaticGateCompiler:
    return EnzymaticGateCompiler()
@pytest.fixture
def nupack_interface() -> NUPACKInterface:
    return NUPACKInterface()
@pytest.fixture
def simple_and_circuit() -> DNACircuitDesign:
    compiler = BitstreamToDNA(method="displacement", seed=42)
    return compiler.compile_network(
        gates=[{"type": "AND", "inputs": ["A", "B"], "output": "C"}],
        input_names=["A", "B"],
        output_names=["C"],
        name="simple_and",
    )
@pytest.fixture
def nand_circuit() -> DNACircuitDesign:
    compiler = BitstreamToDNA(method="displacement", seed=42)
    return compiler.compile_network(
        gates=[
            {"type": "AND", "inputs": ["A", "B"], "output": "X"},
            {"type": "NOT", "inputs": ["X"], "output": "Y"},
        ],
        input_names=["A", "B"],
        output_names=["Y"],
        name="nand_circuit",
    )
@pytest.fixture
def tmp_path_factory_dir(tmp_path: Path) -> Path:
    return tmp_path

__all__ = ['json', 'Path', 'SimpleNamespace', 'Any', 'np', 'pytest', 'dna_mapper', 'BitstreamToDNA', 'CompilationMethod', 'ConcentrationOptimizer', 'CrossHybridizationChecker', 'DNACircuitDesign', 'DNAGate', 'DNAStrand', 'DegradationModel', 'DualRailEncoder', 'EnzymaticGateCompiler', 'GF4ErrorCorrection', 'GateOptimizer', 'GateType', 'HairpinChecker', 'KineticSimulator', 'NUPACKInterface', 'NoiseModel', 'PlateLayout', 'SCNetworkBridge', 'SCPrecisionAnalyzer', 'SequenceDesigner', 'StrandDisplacementCompiler', 'TopologicalAnalyzer', 'estimate_cost', 'export_fasta', 'export_genbank', 'export_json', 'export_nupack_input', 'generate_protocol', 'visualize_circuit', 'visualize_kinetics', '_GC_TARGET_HIGH', '_GC_TARGET_LOW', '_MAX_HOMOPOLYMER', 'designer', 'displacement_compiler', 'enzymatic_compiler', 'nupack_interface', 'simple_and_circuit', 'nand_circuit', 'tmp_path_factory_dir', '__all__']
