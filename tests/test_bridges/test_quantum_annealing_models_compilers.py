# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum-annealing model and compiler tests

"""Exercise validated model contracts and SC problem compilation."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from sc_neurocore.bridges import annealing_backends as backends
from sc_neurocore.bridges.quantum_annealing import (
    CouplerSpec,
    IsingModel,
    ProblemType,
    QUBOModel,
    QubitSpec,
    SCBitstreamQUBO,
    SCToIsing,
    SCToQUBO,
)
from tests.test_bridges.quantum_annealing_test_helpers import (
    simple_adjacency,
    simple_ising,
    unsafe,
)


def test_value_specs_and_problem_types() -> None:
    """Value objects normalize endpoints and expose stable enum values."""
    assert ProblemType.ISING.value == "ising"
    assert ProblemType.QUBO.value == "qubo"
    assert QubitSpec(0, "neuron", 0.5).bias == 0.5
    assert CouplerSpec(2, 1, -1.0) == CouplerSpec(1, 2, -1.0)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: QubitSpec(unsafe(True), "q"), "index"),
        (lambda: QubitSpec(0, ""), "label"),
        (lambda: QubitSpec(0, "q", float("nan")), "bias"),
        (lambda: CouplerSpec(-1, 2), "qubit_a"),
        (lambda: CouplerSpec(1, 1), "distinct"),
        (lambda: CouplerSpec(1, 2, float("inf")), "strength"),
    ],
)
def test_value_specs_reject_invalid_fields(factory: object, match: str) -> None:
    """Invalid indices, labels, and non-finite values fail closed."""
    with pytest.raises(ValueError, match=match):
        unsafe(factory)()


def test_ising_normalizes_and_infers_bounds() -> None:
    """Reverse duplicate couplings merge and omitted size is inferred."""
    model = IsingModel(
        h={2: 0.5},
        J={(2, 0): -1.0, (0, 2): -0.25},
        qubit_labels={0: "zero", 2: "two"},
    )
    assert model.n_qubits == 3
    assert model.J == {(0, 2): -1.25}
    assert model.energy({0: 1, 2: -1}, backend="python") == pytest.approx(0.75)


def test_ising_energy_partial_defaults_and_validation() -> None:
    """Missing spins default to +1 while malformed assignments are rejected."""
    model = simple_ising()
    assert model.energy({0: 1, 1: 1, 2: -1}, backend="python") == pytest.approx(-1.6)
    assert model.energy({}, backend="python") == pytest.approx(-0.6)
    with pytest.raises(ValueError, match="backend"):
        model.energy({}, backend=unsafe("gpu"))
    with pytest.raises(ValueError, match="spin values"):
        model.energy({0: 0})
    with pytest.raises(ValueError, match="spin index"):
        model.energy({unsafe(-1): 1})


def test_large_ising_uses_native_energy_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit native dispatch forwards canonical arrays and the offset."""
    captured: list[object] = []

    def fake_energy(*args: object) -> float:
        captured.extend(args)
        return -7.5

    monkeypatch.setattr(backends, "HAS_RUST_QA", True)
    monkeypatch.setattr(backends, "_rust_ising_energy", fake_energy)
    model = IsingModel(
        h={0: 0.25, 20: -0.5},
        J={(20, 0): -1.25},
        offset=1.0,
        n_qubits=21,
    )
    assert model.energy({0: -1, 20: 1}) == -7.5
    assert captured == [
        [0, 20],
        [0.25, -0.5],
        [0],
        [20],
        [-1.25],
        [-1] + [1] * 20,
        1.0,
    ]


def test_explicit_missing_native_energy_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit Rust request never silently falls back."""
    monkeypatch.setattr(backends, "HAS_RUST_QA", False)
    with pytest.raises(RuntimeError, match="unavailable"):
        simple_ising().energy({}, backend="rust")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"h": {unsafe("x"): 1.0}}, "h index"),
        ({"h": {0: float("nan")}}, "finite"),
        ({"J": {unsafe((0,)): 1.0}}, "two-index"),
        ({"J": {(0, 0): 1.0}}, "distinct"),
        ({"J": {(0, 1): float("inf")}}, "finite"),
        ({"qubit_labels": {unsafe(-1): "x"}}, "label index"),
        ({"qubit_labels": {0: ""}}, "non-empty"),
        ({"qubit_labels": {0: "x", 1: "x"}}, "unique"),
        ({"n_qubits": unsafe(1.5)}, "n_qubits"),
        ({"n_qubits": -1}, "n_qubits"),
        ({"h": {2: 1.0}, "n_qubits": 2}, "smaller"),
        ({"source": unsafe(3)}, "source"),
        ({"offset": float("inf")}, "offset"),
    ],
)
def test_ising_rejects_invalid_models(kwargs: dict[str, object], match: str) -> None:
    """Every malformed structural boundary raises a stable error."""
    with pytest.raises(ValueError, match=match):
        IsingModel(**unsafe(kwargs))


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
