# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chiplet partition-to-routing translation

"""Translate neuron-to-die assignments into directed AER routing tables."""

from __future__ import annotations

from dataclasses import dataclass, field

from sc_neurocore.chiplet.routing import RoutingTable


@dataclass
class PartitionAssignment:
    """Store neuron identifiers grouped by their assigned die."""

    die_assignments: dict[int, list[int]] = field(default_factory=dict)

    def assign(self, neuron_id: int, die_id: int) -> None:
        """Assign ``neuron_id`` to ``die_id``."""
        self.die_assignments.setdefault(die_id, []).append(neuron_id)

    def neurons_on_die(self, die_id: int) -> list[int]:
        """Return neurons assigned to ``die_id``."""
        return self.die_assignments.get(die_id, [])

    def to_routing_tables(
        self, connectivity: list[tuple[int, int, int]]
    ) -> dict[int, RoutingTable]:
        """Convert cross-die connectivity into per-source-die routing tables.

        Parameters
        ----------
        connectivity
            ``(source_neuron, destination_neuron, weight_q88)`` triples.

        Returns
        -------
        dict[int, RoutingTable]
            Tables for source dies with at least one mapped cross-die edge.
        """
        neuron_to_die = {
            neuron_id: die_id
            for die_id, neurons in self.die_assignments.items()
            for neuron_id in neurons
        }
        tables: dict[int, RoutingTable] = {}
        for source, destination, weight in connectivity:
            source_die = neuron_to_die.get(source)
            destination_die = neuron_to_die.get(destination)
            if source_die is None or destination_die is None or source_die == destination_die:
                continue
            table = tables.setdefault(source_die, RoutingTable(die_id=source_die))
            table.add_route(source, destination_die, destination, weight)
        return tables


__all__ = ["PartitionAssignment"]
