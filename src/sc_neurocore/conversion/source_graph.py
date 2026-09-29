# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dense source graph lowering and activation association

"""Lower actual source invocations into a dense IF target's supported operators."""

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import torch
from torch import fx, nn
from torch.nn import functional as F

from .if_parameters import FloatArray
from .model_trace import capture_inference_graph
from .qcfs import QCFSActivation


@dataclass(frozen=True)
class SourceActivation:
    """Source activation kind, exact invocation and learned quantization metadata."""

    kind: Literal["linear", "relu", "qcfs"] = "linear"
    node_name: str = ""
    theta: float = 1.0
    steps: int | None = None


@dataclass(frozen=True)
class SourceLayer:
    """One owned affine coefficient set followed by its actual source activation."""

    weight: FloatArray
    bias: FloatArray | None
    activation: SourceActivation = SourceActivation()


@dataclass(frozen=True)
class SourcePlan:
    """Runnable source inference graph and the supported target's lowered layers."""

    graph: fx.GraphModule
    layers: tuple[SourceLayer, ...]
    max_working_bytes: int


def _output_path(graph: fx.GraphModule) -> list[fx.Node]:
    """Follow the single-tensor data path back to its unique source input."""
    output = next(node for node in graph.graph.nodes if node.op == "output")
    current = output.args[0]
    if not isinstance(current, fx.Node):
        raise ValueError("dense target requires a single tensor output")
    path: list[fx.Node] = []
    while current.op != "placeholder":
        path.append(current)
        predecessor = current.args[0] if current.args else current.kwargs.get("input")
        if not isinstance(predecessor, fx.Node):
            raise ValueError(f"dense target cannot represent operation {current.name}")
        current = predecessor
    return list(reversed(path))


def _source_operation(node: fx.Node, graph: fx.GraphModule) -> nn.Module | SourceActivation | None:
    """Classify one actual call, admitting only explicitly represented operators."""
    if node.op == "call_module" and isinstance(node.target, str):
        if len(node.args) + int("input" in node.kwargs) != 1 or any(
            key != "input" for key in node.kwargs
        ):
            raise ValueError(
                f"dense target cannot represent operation {node.name} with these arguments"
            )
        module = graph.get_submodule(node.target)
        if isinstance(module, nn.Linear):
            return module
        if isinstance(module, QCFSActivation):
            theta = float(module.theta.item())
            if not np.isfinite(theta) or theta <= 0:
                raise ValueError("QCFS theta must be finite and positive")
            if type(module.T) is not int or module.T < 1:
                raise ValueError("QCFS T must be a positive integer")
            return SourceActivation("qcfs", node.name, theta, module.T)
        if isinstance(module, (nn.ReLU, nn.ReLU6)):
            return SourceActivation("relu", node.name, 6.0 if isinstance(module, nn.ReLU6) else 1.0)
        if isinstance(
            module,
            (
                nn.Identity,
                nn.Dropout,
                nn.Dropout1d,
                nn.Dropout2d,
                nn.Dropout3d,
                nn.AlphaDropout,
                nn.FeatureAlphaDropout,
            ),
        ):
            return None
        if isinstance(module, nn.Flatten) and module.start_dim == 1 and module.end_dim == -1:
            return None
    if node.op == "call_function" and node.target is F.dropout:
        training = node.kwargs.get("training", node.args[2] if len(node.args) > 2 else True)
        probability = node.kwargs.get("p", node.args[1] if len(node.args) > 1 else 0.5)
        if training is False or probability == 0:
            return None
    if node.op == "call_function" and node.target in (torch.relu, F.relu, F.relu6):
        return SourceActivation("relu", node.name, 6.0 if node.target is F.relu6 else 1.0)
    if node.op == "call_method" and node.target in ("relu", "relu_"):
        return SourceActivation("relu", node.name)
    raise ValueError(f"dense target cannot represent operation {node.name} ({node.target})")


def compile_source_graph(model: nn.Module, max_working_bytes: int) -> SourcePlan:
    """Lower a supported straight-line source graph using actual execution order.

    Parameters
    ----------
    model : nn.Module
        Source network. Repeated/shared calls remain distinct invocations.
    max_working_bytes : int
        Budget for independent source storage and inserted identity coefficients.

    Returns
    -------
    SourcePlan
        Owned dense coefficients with per-invocation activation metadata.

    Raises
    ------
    ValueError
        If inputs, outputs or operators cannot be represented by the dense target.
    MemoryError
        If source copying or inserted identity storage exceeds the budget.

    Notes
    -----
    Consecutive affine maps are composed without inserting an IF nonlinearity.
    Composition uses float64 arithmetic; source/target loss still needs measured
    acceptance, including differences from the source dtype's rounding order.
    """
    graph = capture_inference_graph(model, max_working_bytes)
    layers: list[SourceLayer] = []
    for node in _output_path(graph):
        operation = _source_operation(node, graph)
        if isinstance(operation, nn.Linear):
            if operation.weight.is_complex() or (
                operation.bias is not None and operation.bias.is_complex()
            ):
                raise ValueError("dense source coefficients must be real")
            weight: FloatArray = operation.weight.detach().cpu().double().numpy().copy()
            bias = (
                None
                if operation.bias is None
                else operation.bias.detach().cpu().double().numpy().copy()
            )
            if layers and layers[-1].activation.kind == "linear":
                previous = layers.pop()
                if 16 * weight.shape[0] * previous.weight.shape[1] > max_working_bytes:
                    raise MemoryError("fused affine exceeds working byte budget")
                if previous.bias is not None:
                    bias = weight @ previous.bias if bias is None else weight @ previous.bias + bias
                weight = weight @ previous.weight
            layers.append(SourceLayer(weight, bias))
        elif isinstance(operation, SourceActivation):
            if not layers:
                raise ValueError("No Linear layer precedes the source activation")
            if layers[-1].activation.kind != "linear":
                width = layers[-1].weight.shape[0]
                if 16 * width * width > max_working_bytes:
                    raise MemoryError("activation identity exceeds working byte budget")
                layers.append(SourceLayer(np.eye(width), None))
            layers[-1] = replace(layers[-1], activation=operation)
    if not layers:
        raise ValueError("No Linear layers found in source graph")
    return SourcePlan(graph, tuple(layers), max_working_bytes)
