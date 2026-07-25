# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic-bitstream QUBO compiler tests

"""Validate stochastic-bitstream weight and pruning QUBO formulations."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from sc_neurocore.bridges.quantum_annealing import SCBitstreamQUBO
from tests.test_bridges.quantum_annealing_test_helpers import unsafe


def test_bitstream_weight_optimization_matches_objective() -> None:
    """Weight-selection QUBO reproduces the least-squares objective."""
    target = np.array([0.5, 0.3, 0.8])
    candidates = np.array([[0.4, 0.3, 0.7], [0.6, 0.2, 0.9], [0.5, 0.3, 0.8]])
    model = SCBitstreamQUBO().weight_optimization(target, candidates, n_bits=3)
    for bits_tuple in product((0, 1), repeat=3):
        bits = dict(enumerate(bits_tuple))
        vector = np.asarray(bits_tuple, dtype=np.float64)
        expected = float(np.sum((target - candidates @ vector) ** 2))
        assert model.energy(bits) == pytest.approx(expected)


def test_bitstream_weight_optimization_omits_zero_cross_terms() -> None:
    """Orthogonal candidate columns do not create zero-valued QUBO couplings."""
    model = SCBitstreamQUBO().weight_optimization(
        np.array([1.0, 1.0]),
        np.eye(2),
        n_bits=2,
    )
    assert (0, 1) not in model.Q


def test_bitstream_pruning_encodes_exact_cardinality() -> None:
    """Pruning includes reverse-only edges and symmetric importance."""
    adjacency = np.array([[0.0, 0.0, 0.9], [0.1, 0.0, 0.8], [0.9, 0.8, 0.0]])
    importance = np.array([[0.0, 0.2, 0.9], [0.4, 0.0, 0.8], [0.9, 0.8, 0.0]])
    model = SCBitstreamQUBO(penalty=5.0).pruning(adjacency, importance, 2)
    assert model.n_qubits == 3
    assert model.offset == 20.0
    assert model.Q[(0, 0)] == pytest.approx(-0.3 - 15.0)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: SCBitstreamQUBO(0.0), "penalty"),
        (
            lambda: SCBitstreamQUBO().weight_optimization(np.ones((1, 1)), np.ones((1, 1))),
            "one-dimensional",
        ),
        (lambda: SCBitstreamQUBO().weight_optimization(np.ones(2), np.ones((3, 2))), "one row"),
        (
            lambda: SCBitstreamQUBO().weight_optimization(np.array([np.nan]), np.ones((1, 1))),
            "finite",
        ),
        (
            lambda: SCBitstreamQUBO().weight_optimization(
                np.ones(1), np.ones((1, 1)), unsafe(True)
            ),
            "positive",
        ),
        (lambda: SCBitstreamQUBO().weight_optimization(np.ones(1), np.ones((1, 1)), 2), "exceed"),
        (lambda: SCBitstreamQUBO().pruning(np.eye(2), np.ones((3, 3)), 0), "match"),
        (
            lambda: SCBitstreamQUBO().pruning(np.eye(2), np.array([[0.0, np.inf], [0.0, 0.0]]), 0),
            "finite",
        ),
        (
            lambda: SCBitstreamQUBO().pruning(np.ones((2, 2)), np.ones((2, 2)), unsafe(1.5)),
            "integer",
        ),
        (lambda: SCBitstreamQUBO().pruning(np.eye(2), np.zeros((2, 2)), 1), "between"),
    ],
)
def test_bitstream_compiler_rejects_invalid_inputs(call: object, match: str) -> None:
    """SC-specific QUBOs reject impossible or malformed formulations."""
    with pytest.raises(ValueError, match=match):
        unsafe(call)()
