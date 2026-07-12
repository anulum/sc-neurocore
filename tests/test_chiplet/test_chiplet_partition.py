# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chiplet partition translation contracts

"""Tests for neuron-to-die assignment and AER routing translation."""

from __future__ import annotations

from sc_neurocore.chiplet import PartitionAssignment


def test_assignment_queries_preserve_insertion_order() -> None:
    assignment = PartitionAssignment()
    assignment.assign(0, 0)
    assignment.assign(1, 0)
    assignment.assign(2, 1)
    assert assignment.neurons_on_die(0) == [0, 1]
    assert assignment.neurons_on_die(1) == [2]
    assert assignment.neurons_on_die(99) == []


def test_local_and_unmapped_edges_do_not_create_routes() -> None:
    assignment = PartitionAssignment({0: [0, 1]})
    assert assignment.to_routing_tables([(0, 1, 256), (0, 99, 256)]) == {}


def test_cross_die_edges_create_source_table_only() -> None:
    assignment = PartitionAssignment({0: [0, 1], 1: [4, 5]})
    tables = assignment.to_routing_tables([(0, 4, 256), (1, 5, 128), (0, 1, 256), (4, 5, 256)])
    assert set(tables) == {0}
    assert tables[0].num_entries == 2
    assert [entry.dst_die for entry in tables[0].entries] == [1, 1]
