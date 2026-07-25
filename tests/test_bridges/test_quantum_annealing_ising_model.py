# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum-annealing Ising model tests

"""Validate Ising normalization, energy dispatch, and malformed inputs."""

from __future__ import annotations

import pytest

from sc_neurocore.bridges import annealing_backends as backends
from sc_neurocore.bridges.quantum_annealing import IsingModel
from tests.test_bridges.quantum_annealing_test_helpers import simple_ising, unsafe


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
