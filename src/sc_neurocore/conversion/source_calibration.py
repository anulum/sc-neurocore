# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Per-invocation source graph calibration

"""Measure module and functional activations at their actual inference nodes."""

import numpy as np
import torch
from torch import fx, nn

from .random_custody import preserved_random_state
from .source_graph import SourcePlan


class ActivationMeasurement(fx.Interpreter):
    """Retain independent observed tensors for the requested source invocations."""

    def __init__(self, graph: fx.GraphModule, names: set[str]) -> None:
        """Initialize actual graph execution and an empty observation map."""
        super().__init__(graph)
        self.names = names
        self.observed: dict[str, torch.Tensor] = {}

    def run_node(self, n: fx.Node) -> object:
        """Snapshot each selected actual tensor before subsequent in-place operators."""
        output: object = super().run_node(n)
        if n.name in self.names:
            if not isinstance(output, torch.Tensor):
                raise ValueError("calibration activation must be a tensor")
            self.observed[n.name] = output.detach().cpu().clone()
        return output


def calibrate_source_nodes(
    plan: SourcePlan, data: torch.Tensor, percentile: float
) -> dict[str, float]:
    """Measure source ReLU invocation thresholds with inference graph execution.

    Parameters
    ----------
    plan : SourcePlan
        Independent source graph and actual activation metadata.
    data : Tensor
        Nonempty finite source-format calibration input.
    percentile : float
        Finite percentile in the closed interval zero to one hundred.

    Returns
    -------
    dict of str to float
        Positive scales indexed by actual activation node name.

    Raises
    ------
    MemoryError
        If source input validation and activation buffers exceed the plan budget.
    ValueError
        If inputs, percentile or observed tensors are invalid.
    """
    if not 0 <= percentile <= 100:
        raise ValueError("calibration percentile must be between zero and 100")
    if data.numel() == 0:
        raise ValueError("calibration data must be nonempty and finite")
    rows = data.numel() // max(1, int(data.shape[-1])) if data.ndim else 1
    widths = sum(
        module.out_features
        for node in plan.graph.graph.nodes
        if node.op == "call_module" and isinstance(node.target, str)
        for module in (plan.graph.get_submodule(node.target),)
        if isinstance(module, nn.Linear)
    )
    if 8 * (2 * data.numel() + 8 * rows * widths) > plan.max_working_bytes:
        raise MemoryError("source calibration buffers exceed working byte budget")
    if not bool(torch.isfinite(data).all()):
        raise ValueError("calibration data must be nonempty and finite")
    names = {layer.activation.node_name for layer in plan.layers if layer.activation.kind == "relu"}
    measurement = ActivationMeasurement(plan.graph, names)
    with preserved_random_state(), torch.no_grad():
        measurement.run(data)
    result: dict[str, float] = {}
    for name, activation in measurement.observed.items():
        if activation.numel() == 0 or not bool(torch.isfinite(activation).all()):
            raise ValueError("calibration activation must be finite and nonempty")
        result[name] = max(float(np.percentile(activation.double().numpy(), percentile)), 1e-6)
    return result
