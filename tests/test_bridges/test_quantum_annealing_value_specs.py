# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum-annealing value specification tests

"""Validate quantum-annealing value objects and problem-type identifiers."""

from __future__ import annotations

import pytest

from sc_neurocore.bridges.quantum_annealing import CouplerSpec, ProblemType, QubitSpec
from tests.test_bridges.quantum_annealing_test_helpers import unsafe


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
