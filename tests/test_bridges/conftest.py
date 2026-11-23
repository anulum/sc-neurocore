# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Fixtures shared by the bridge tests

"""Fixtures shared by the DNA-mapper and photonic NoC bridge tests."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from sc_neurocore.bridges.dna_mapper import (
    BitstreamToDNA,
    DNACircuitDesign,
    EnzymaticGateCompiler,
    NUPACKInterface,
    SequenceDesigner,
    StrandDisplacementCompiler,
)
from sc_neurocore.bridges.photonic_noc import PhotonicCircuitDesign, SCToPhotonic


@pytest.fixture
def designer() -> SequenceDesigner:
    """Return a sequence designer seeded with 42."""
    return SequenceDesigner(seed=42)


@pytest.fixture
def displacement_compiler() -> StrandDisplacementCompiler:
    """Return a strand-displacement gate compiler."""
    return StrandDisplacementCompiler()


@pytest.fixture
def enzymatic_compiler() -> EnzymaticGateCompiler:
    """Return an enzymatic gate compiler."""
    return EnzymaticGateCompiler()


@pytest.fixture
def nupack_interface() -> NUPACKInterface:
    """Return a NUPACK interface."""
    return NUPACKInterface()


@pytest.fixture
def simple_and_circuit() -> DNACircuitDesign:
    """Return a compiled one-gate AND circuit, C = A AND B."""
    compiler = BitstreamToDNA(method="displacement", seed=42)
    return compiler.compile_network(
        gates=[{"type": "AND", "inputs": ["A", "B"], "output": "C"}],
        input_names=["A", "B"],
        output_names=["C"],
        name="simple_and",
    )


@pytest.fixture
def nand_circuit() -> DNACircuitDesign:
    """Return a compiled NAND circuit, an AND gate followed by a NOT gate."""
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
def simple_adjacency() -> NDArray[np.float64]:
    """Return the weighted adjacency of a 4-node mesh network."""
    return np.array(
        [
            [0.0, 1.0, 0.5, 0.0],
            [1.0, 0.0, 0.0, 0.8],
            [0.5, 0.0, 0.0, 1.0],
            [0.0, 0.8, 1.0, 0.0],
        ]
    )


@pytest.fixture
def simple_design(simple_adjacency: NDArray[np.float64]) -> PhotonicCircuitDesign:
    """Return the 4-node mesh compiled to a photonic design."""
    compiler = SCToPhotonic()
    return compiler.compile(simple_adjacency, name="test_noc")
