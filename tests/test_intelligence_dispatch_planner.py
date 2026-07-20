# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Heterogeneous dispatch planner contracts

"""Contracts for heterogeneous compiler dispatch planning."""

from sc_neurocore.compiler.intelligence import plan_heterogeneous_dispatch


def test_dispatch_defaults_to_fpga_for_empty_backends() -> None:
    """An empty backend list defaults to a single FPGA backend."""
    plan = plan_heterogeneous_dispatch({"a": "x + 1"}, [])

    assert "fpga" in plan.backends


def test_dispatch_assigns_remainder_neurons_to_first_backend() -> None:
    """Indivisible neuron counts preserve the requested total."""
    plan = plan_heterogeneous_dispatch(
        {"a": "x + 1", "b": "y + 1", "c": "z + 1"},
        ["fpga", "asic"],
        neuron_count=1001,
    )

    assert sum(plan.total_neurons_per_backend.values()) == 1001
