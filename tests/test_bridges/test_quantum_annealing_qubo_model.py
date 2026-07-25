# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum-annealing QUBO model tests

"""Validate QUBO normalization, energy, conversion, and malformed inputs."""

from __future__ import annotations

from itertools import product

import pytest

from sc_neurocore.bridges.quantum_annealing import QUBOModel
from tests.test_bridges.quantum_annealing_test_helpers import unsafe


def test_qubo_normalization_energy_and_exact_ising_conversion() -> None:
    """QUBO canonicalization preserves energy under the spin transform."""
    qubo = QUBOModel(
        Q={(0, 0): -2.0, (1, 1): 1.5, (1, 0): 0.75, (0, 1): 0.25},
        offset=0.3,
        qubit_labels={0: "a", 1: "b"},
        source="qubo",
    )
    assert qubo.Q[(0, 1)] == 1.0
    ising = qubo.to_ising()
    for bits_tuple in product((0, 1), repeat=2):
        bits = dict(enumerate(bits_tuple))
        spins = {index: 2 * bit - 1 for index, bit in bits.items()}
        assert ising.energy(spins, backend="python") == pytest.approx(qubo.energy(bits))
    assert ising.source == "qubo (QUBO→Ising)"


def test_qubo_energy_validates_bits() -> None:
    """QUBO assignments accept only non-negative binary entries."""
    model = QUBOModel(Q={(0, 0): -1.0})
    assert model.energy({0: 1}) == -1.0
    with pytest.raises(ValueError, match="bit values"):
        model.energy({0: -1})
    with pytest.raises(ValueError, match="bit index"):
        model.energy({unsafe(True): 1})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"Q": {unsafe((0,)): 1.0}}, "two-index"),
        ({"Q": {(unsafe("x"), 0): 1.0}}, "Q index"),
        ({"Q": {(0, 0): float("nan")}}, "finite"),
        ({"qubit_labels": {0: ""}}, "non-empty"),
        ({"qubit_labels": {0: "x", 1: "x"}}, "unique"),
        ({"n_qubits": unsafe(False)}, "n_qubits"),
        ({"n_qubits": -2}, "n_qubits"),
        ({"Q": {(2, 2): 1.0}, "n_qubits": 2}, "smaller"),
        ({"source": unsafe(None)}, "source"),
        ({"offset": float("-inf")}, "offset"),
    ],
)
def test_qubo_rejects_invalid_models(kwargs: dict[str, object], match: str) -> None:
    """Malformed QUBO structure fails at construction."""
    with pytest.raises(ValueError, match=match):
        QUBOModel(**unsafe(kwargs))
