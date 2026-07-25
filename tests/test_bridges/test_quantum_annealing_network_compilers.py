# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum-annealing network compiler tests

"""Validate stochastic-network compilation to Ising and QUBO models."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.bridges.quantum_annealing import SCToIsing, SCToQUBO
from tests.test_bridges.quantum_annealing_test_helpers import simple_adjacency, unsafe


def test_sc_to_ising_compiles_signs_labels_and_biases() -> None:
    """SC compilation symmetrizes weights and applies configured scales."""
    adjacency = simple_adjacency()
    model = SCToIsing(coupling_scale=2.0, field_scale=1.0).compile(
        adjacency,
        ["X", "Y", "Z"],
        np.array([0.5, -0.3, 0.0]),
        "network",
    )
    assert model.h == {0: 0.5, 1: -0.3, 2: 0.0}
    assert model.J == {(0, 1): -2.0, (1, 2): -2.0}
    assert model.qubit_labels[0] == "X"
    assert model.source == "network"

    inhibitory = SCToIsing().compile(np.array([[0.0, -1.0], [-1.0, 0.0]]))
    assert inhibitory.J[(0, 1)] > 0.0


def test_sc_to_qubo_compiles_diagonal_and_couplings() -> None:
    """QUBO compilation produces a canonical labeled matrix."""
    model = SCToQUBO(penalty=3.0).compile(simple_adjacency(), name="qubo")
    assert model.n_qubits == 3
    assert model.Q[(0, 0)] == -1.0
    assert model.Q[(0, 1)] == 3.0
    assert model.source == "qubo"


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: SCToIsing(float("nan")), "coupling_scale"),
        (lambda: SCToIsing(field_scale=float("inf")), "field_scale"),
        (lambda: SCToQUBO(0.0), "penalty"),
        (lambda: SCToIsing().compile(np.ones((2, 3))), "square"),
        (lambda: SCToIsing().compile(np.empty((0, 0))), "square"),
        (lambda: SCToIsing().compile(np.array([[0.0, np.nan], [0.0, 0.0]])), "finite"),
        (lambda: SCToIsing().compile(np.eye(2), ["only"]), "exactly"),
        (lambda: SCToIsing().compile(np.eye(2), ["x", "x"]), "unique"),
        (lambda: SCToIsing().compile(np.eye(2), ["x", ""]), "non-empty"),
        (lambda: SCToIsing().compile(np.eye(2), biases=np.ones(3)), "biases"),
        (lambda: SCToIsing().compile(np.eye(2), biases=np.array([0.0, np.inf])), "finite"),
        (lambda: SCToIsing().compile(np.eye(2), name=""), "name"),
        (lambda: SCToQUBO().compile(np.eye(2), name=" "), "name"),
    ],
)
def test_network_compilers_reject_invalid_inputs(call: object, match: str) -> None:
    """Compiler shape, label, scale, and finiteness boundaries fail closed."""
    with pytest.raises(ValueError, match=match):
        unsafe(call)()
