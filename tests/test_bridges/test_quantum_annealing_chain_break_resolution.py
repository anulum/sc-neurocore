# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — quantum-annealing chain-break contracts

from __future__ import annotations


import pytest

from sc_neurocore.bridges.quantum_annealing import (
    ChainBreakResolver,
    IsingModel,
)
from tests.test_bridges.quantum_annealing_test_helpers import unsafe


def test_chain_resolution_and_break_statistics() -> None:
    """Majority voting and break analysis handle ties and single-qubit chains."""
    samples = [{0: 1, 1: 1, 2: -1}, {0: 1, 1: -1, 2: -1}]
    chains = {0: [0, 1], 1: [2]}
    resolved = ChainBreakResolver().resolve(samples, chains)
    assert resolved == [{0: 1, 1: -1}, {0: 1, 1: -1}]
    stats = ChainBreakResolver().analyze_breaks(samples, chains)
    assert stats["total_breaks"] == 1
    assert stats["break_rate"] == 0.5
    assert stats["per_chain"] == {0: 0.5, 1: 0.0}
    assert ChainBreakResolver().analyze_breaks([], chains)["break_rate"] == 0.0


def test_chain_energy_minimization_refines_vote() -> None:
    """Energy minimization flips a voted spin only when energy decreases."""
    model = IsingModel(h={0: 2.0, 1: -2.0}, n_qubits=2)
    result = ChainBreakResolver("minimize_energy").resolve(
        [{0: 1, 1: 1}],
        {0: [0], 1: [1]},
        model,
    )
    assert result == [{0: -1, 1: 1}]


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: ChainBreakResolver("bad"), "Unknown method"),
        (lambda: ChainBreakResolver("minimize_energy").resolve([], {}), "model is required"),
        (lambda: ChainBreakResolver().resolve(unsafe("bad"), {}), "sequence"),
        (lambda: ChainBreakResolver().resolve([{unsafe(-1): 1}], {}), "indices"),
        (lambda: ChainBreakResolver().resolve([{0: 0}], {}), "spins"),
        (lambda: ChainBreakResolver().resolve([], {unsafe(-1): [0]}), "logical"),
        (lambda: ChainBreakResolver().resolve([], {0: []}), "non-empty"),
        (lambda: ChainBreakResolver().resolve([], {0: [unsafe("x")]}), "physical"),
        (lambda: ChainBreakResolver().resolve([], {0: [1, 1]}), "duplicate"),
        (lambda: ChainBreakResolver().resolve([], {0: [1], 1: [1]}), "multiple"),
        (lambda: ChainBreakResolver().resolve([], {0: [1]}, unsafe("bad")), "IsingModel"),
        (lambda: ChainBreakResolver().resolve([], {2: [1]}, IsingModel(h={0: 0.0})), "fit within"),
    ],
)
def test_chain_resolution_rejects_invalid_inputs(call: object, match: str) -> None:
    """Malformed samples, chains, and model mappings fail closed."""
    with pytest.raises(ValueError, match=match):
        unsafe(call)()
