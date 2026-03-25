# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SNN computation graph optimizer

"""LLVM-style optimization passes for SNN computation graphs."""

from .passes import (
    SNNGraph,
    LayerNode,
    dead_neuron_elimination,
    layer_fusion,
    redundancy_elimination,
    optimize,
    OptimizationReport,
)

__all__ = [
    "SNNGraph",
    "LayerNode",
    "dead_neuron_elimination",
    "layer_fusion",
    "redundancy_elimination",
    "optimize",
    "OptimizationReport",
]
