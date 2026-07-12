# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CPPN developmental genome encoding

"""Generate structured connection weights from CPPN genomes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, List

import numpy as np


class ActivationFunc(Enum):
    """Select the nonlinear response used by a CPPN node."""

    SIN = "sin"
    GAUSS = "gauss"
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    STEP = "step"


@dataclass
class CPPNNode:
    """One node in a CPPN network."""

    node_id: int
    activation: ActivationFunc = ActivationFunc.LINEAR
    bias: float = 0.0


@dataclass
class CPPNEdge:
    """One edge in a CPPN network."""

    src: int
    dst: int
    weight: float = 1.0
    enabled: bool = True


class CPPNGenome:
    """Compositional Pattern Producing Network for developmental encoding."""

    def __init__(self) -> None:
        self.nodes: List[CPPNNode] = [
            CPPNNode(0, ActivationFunc.LINEAR),  # input x
            CPPNNode(1, ActivationFunc.LINEAR),  # input y
            CPPNNode(2, ActivationFunc.SIGMOID),  # output
        ]
        self.edges: List[CPPNEdge] = [
            CPPNEdge(0, 2, 1.0),
            CPPNEdge(1, 2, 1.0),
        ]

    def query(self, x: float, y: float) -> float:
        """Query the CPPN at coordinates (x, y)."""
        values = {0: x, 1: y}
        for node in self.nodes[2:]:
            total = node.bias
            for edge in self.edges:
                if edge.dst == node.node_id and edge.enabled and edge.src in values:
                    total += edge.weight * values[edge.src]
            values[node.node_id] = self._activate(total, node.activation)
        return values.get(2, 0.0)

    @staticmethod
    def _activate(x: float, func: ActivationFunc) -> float:
        if func == ActivationFunc.SIN:
            return float(np.sin(x))
        if func == ActivationFunc.GAUSS:
            return float(np.exp(-x * x))
        if func == ActivationFunc.SIGMOID:
            return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))
        if func == ActivationFunc.STEP:
            return 1.0 if x > 0 else 0.0
        return float(x)  # LINEAR

    def generate_weight_matrix(self, rows: int, cols: int) -> np.ndarray[Any, Any]:
        """Generate a weight matrix by querying CPPN at grid positions."""
        w = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                x = 2.0 * r / max(1, rows - 1) - 1.0
                y = 2.0 * c / max(1, cols - 1) - 1.0
                w[r, c] = self.query(x, y)
        return w

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the CPPN graph."""
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the CPPN graph."""
        return len(self.edges)


__all__ = ["ActivationFunc", "CPPNEdge", "CPPNGenome", "CPPNNode"]
