# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hardware-targeted neuron graph contracts

"""Define the hardware-targeted graph records consumed by FPGA compilation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

DelaySteps = int | tuple[int, ...]


@dataclass
class NeuronSpec:
    """Describe one neuron population in the compiled graph.

    Parameters
    ----------
    name : str
        Unique population identifier matching the NIR node name.
    neuron_type : str
        Canonical neuron type such as ``"lif"``, ``"if"``, ``"li"``,
        ``"cuba_lif"``, or ``"cuba_li"``.
    n_neurons : int
        Number of neurons in the population.
    params : dict[str, numpy.ndarray]
        Canonical neuron parameters stored as arrays.
    dt : float
        Simulation timestep inherited from NIR import.
    """

    name: str
    neuron_type: str
    n_neurons: int
    params: dict[str, np.ndarray[Any, Any]] = field(default_factory=dict)
    dt: float = 1.0


@dataclass
class ConnectionSpec:
    """Describe a weighted edge between neuron populations.

    Parameters
    ----------
    src : str
        Source population name.
    dst : str
        Destination population name.
    weights : numpy.ndarray
        Weight matrix with shape ``(n_dst, n_src)``.
    bias : numpy.ndarray or None
        Optional destination bias vector with shape ``(n_dst,)``.
    delay_steps : int or tuple[int, ...]
        Scalar delay or one explicit delay per source column.
    source_threshold : numpy.ndarray or None
        Optional threshold applied before the weight matrix.
    destination_threshold : numpy.ndarray or None
        Optional threshold applied after affine accumulation.
    """

    src: str
    dst: str
    weights: np.ndarray[Any, Any]
    bias: np.ndarray[Any, Any] | None = None
    delay_steps: DelaySteps = 0
    source_threshold: np.ndarray[Any, Any] | None = None
    destination_threshold: np.ndarray[Any, Any] | None = None


@dataclass(frozen=True, slots=True)
class HierarchyInstanceSpec:
    """Preserve provenance for a nested graph flattened into hardware IR.

    Parameters
    ----------
    instance_id : str
        Parent-graph node name of the nested graph instance.
    module_name : str
        Stable HDL module identifier assigned to the instance boundary.
    node_name_prefix : str
        Namespace prefix applied to the nested nodes during flattening.
    """

    instance_id: str
    module_name: str
    node_name_prefix: str


@dataclass
class NeuronGraph:
    """Describe a complete network ready for FPGA compilation.

    Parameters
    ----------
    populations : list[NeuronSpec]
        Populations in deterministic topological order.
    connections : list[ConnectionSpec]
        Weighted connections between populations.
    input_pop : str
        Input boundary or first population name.
    output_pop : str
        Output boundary or final population name.
    dt : float
        Global simulation timestep.
    hierarchy : tuple[HierarchyInstanceSpec, ...]
        Nested instances flattened for hardware lowering.
    """

    populations: list[NeuronSpec]
    connections: list[ConnectionSpec]
    input_pop: str
    output_pop: str
    dt: float = 1.0
    hierarchy: tuple[HierarchyInstanceSpec, ...] = ()

    @property
    def total_neurons(self) -> int:
        """Return the neuron count across all populations."""
        return sum(population.n_neurons for population in self.populations)

    @property
    def total_synapses(self) -> int:
        """Return the matrix-entry count across all connections."""
        return sum(connection.weights.size for connection in self.connections)

    @property
    def neuron_types(self) -> set[str]:
        """Return the canonical neuron types present in the graph."""
        return {population.neuron_type for population in self.populations}

    def summary(self) -> str:
        """Return a deterministic human-readable graph summary."""
        lines = [
            f"NeuronGraph: {len(self.populations)} populations, "
            f"{len(self.connections)} connections",
            f"  Total neurons:  {self.total_neurons}",
            f"  Total synapses: {self.total_synapses}",
            f"  Neuron types:   {', '.join(sorted(self.neuron_types))}",
            f"  Input:  {self.input_pop}",
            f"  Output: {self.output_pop}",
            f"  dt: {self.dt}",
            "",
            "  Populations:",
        ]
        for population in self.populations:
            lines.append(
                f"    {population.name}: {population.neuron_type} × {population.n_neurons}"
            )
        lines.extend(("", "  Connections:"))
        for connection in self.connections:
            shape = f"{connection.weights.shape[1]}→{connection.weights.shape[0]}"
            bias = " +bias" if connection.bias is not None else ""
            delay = f" delay={connection.delay_steps}" if connection.delay_steps else ""
            lines.append(f"    {connection.src} → {connection.dst}: {shape}{bias}{delay}")
        return "\n".join(lines)
