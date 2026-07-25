# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — quantum-annealing hardware-graph contracts

from __future__ import annotations


import pytest

from sc_neurocore.bridges.quantum_annealing import (
    HardwareGraph,
)
from tests.test_bridges.quantum_annealing_test_helpers import simple_ising, unsafe


def test_hardware_graph_capacities_and_embedding() -> None:
    """Each supported topology reports its documented idealized capacity."""
    assert HardwareGraph("chimera", 2).n_physical_qubits == 32
    assert HardwareGraph("pegasus", 16).n_physical_qubits == 5760
    assert HardwareGraph("zephyr", 2).n_physical_qubits == 192
    graph = HardwareGraph("pegasus", 2)
    result = graph.can_embed(simple_ising())
    assert result["embeddable"] is True
    assert graph.connectivity == 15
    assert result["utilization_pct"] > 0.0


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: HardwareGraph("unknown"), "Unknown topology"),
        (lambda: HardwareGraph("chimera", 0), "positive"),
        (lambda: HardwareGraph("pegasus", 1), "at least two"),
        (lambda: HardwareGraph().can_embed(unsafe("bad")), "non-empty"),
    ],
)
def test_hardware_graph_rejects_invalid_inputs(call: object, match: str) -> None:
    """Unknown topology, invalid size, and empty models are rejected."""
    with pytest.raises(ValueError, match=match):
        unsafe(call)()
