# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Explicit snnTorch inference interoperability

"""Convert supported inference definitions without silently discarding SC contracts.

The qualified numerical lane is CPU float64, one vector per timestep. It uses
real snnTorch cells and ordinary Torch linear layers. Euler and decay-form
updates can round differently; this is not a bitstream or training checkpoint
format. Stochastic stream contracts are deliberately refused.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from importlib import import_module
import json
from typing import TYPE_CHECKING, Any, get_args

import nir
import numpy as np
import snntorch as snn
from snntorch import surrogate
import torch
from torch import nn

from sc_neurocore.ir.scnir_schema import (
    SCNIRDocument,
    SCNIRStream,
    SCNIRPrecision,
    SCNIRSource,
    SCNIRCorrelationConstraint,
    SCNIRStreamTransform,
    SCNIRHierarchyInstance,
    SCNIRHierarchyPort,
    SCNIREncoding,
    SCNIRRounding,
    SCNIROverflow,
    SCNIRSourceKind,
)

from .execution_metadata import read_execution_settings, write_execution_metadata
from .graph_identity import graph_order
from .parser import SCNetwork, from_nir
from .export import to_nir

nirtorch = import_module("nirtorch")

if TYPE_CHECKING:

    class _ModuleBase:
        def __init__(self) -> None: ...
        def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
        def register_buffer(self, name: str, tensor: Any) -> None: ...

else:
    _ModuleBase = nn.Module


def _refuse_streams(scnir: object | None) -> None:
    if scnir is not None:
        raise ValueError(
            "snnTorch float inference cannot preserve SC bitstream length, encoding, "
            "Q-format, source randomness or correlation constraints; keep the SC-NIR document"
        )


def _refuse_metadata(value: object, seen: set[int] | None = None) -> None:
    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(
        value,
        (
            SCNIRDocument,
            SCNIRStream,
            SCNIRPrecision,
            SCNIRSource,
            SCNIRCorrelationConstraint,
            SCNIRStreamTransform,
            SCNIRHierarchyInstance,
            SCNIRHierarchyPort,
        ),
    ):
        _refuse_streams(value)
    elif isinstance(value, Mapping):
        schema = value.get("schema_version")
        encoding = value.get("encoding")
        rounding = value.get("rounding")
        overflow = value.get("overflow")
        source_kind = value.get("kind")
        if (
            isinstance(schema, str)
            and schema.startswith("sc-neurocore.scnir.")
            or set(value)
            & {"bitstream_length", "q_format", "qformat", "fractional_bits", "accumulator_bits"}
            or isinstance(encoding, str)
            and encoding in get_args(SCNIREncoding)
            or isinstance(rounding, str)
            and rounding in get_args(SCNIRRounding)
            or isinstance(overflow, str)
            and overflow in get_args(SCNIROverflow)
            or isinstance(source_kind, str)
            and source_kind in get_args(SCNIRSourceKind)
        ):
            _refuse_streams(value)
        for item in value.values():
            _refuse_metadata(item, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _refuse_metadata(item, seen)
    elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            decoded: object = json.loads(value)
        except (ValueError, RecursionError):
            return
        _refuse_metadata(decoded)


def _vector64(sample: Any) -> None:
    if not isinstance(sample, torch.Tensor):
        raise TypeError("Expected a Torch tensor")
    if sample.device.type != "cpu" or sample.dtype != torch.float64:
        raise ValueError("The qualified snnTorch interchange profile requires CPU float64")
    if sample.ndim != 1 or sample.numel() == 0 or not bool(torch.isfinite(sample).all()):
        raise ValueError("Provide one nonempty finite vector per timestep; batches are unsupported")


def _parameter(value: Any, width: int, label: str) -> np.ndarray[Any, Any]:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{label} must be a Torch tensor")
    _vector64(value.reshape(-1))
    try:
        return np.broadcast_to(value.detach().numpy().reshape(-1), (width,)).copy()
    except ValueError as exc:
        raise ValueError(f"{label} must be scalar or have one value per neuron") from exc


def from_snntorch(
    model: Any, sample: Any, *, dt: float = 1e-4, scnir: object | None = None
) -> SCNetwork:
    """Import a sequential snnTorch inference definition through public NIRTorch FX.

    Parameters
    ----------
    model : torch.nn.Sequential
        Flat sequence of Linear, Leaky and Synaptic layers, CPU float64.
        Cells must use immediate, uniform zero or subtract reset, hidden state
        and spike-only outputs. Source parameters and live state are untouched.
    sample : torch.Tensor
        Finite unbatched input vector defining the boundary width.
    dt : float
        Positive finite timestep in the intended NIR time unit.
    scnir : object or None
        An attached stochastic stream contract is refused rather than erased.

    Returns
    -------
    SCNetwork
        Fresh-state inference graph with recorded timestep and reset convention.

    Raises
    ------
    TypeError
        The source is not a supported Torch sequence or tensor.
    ValueError
        Precision, dynamics, topology or an SC contract cannot be preserved.
    """
    _refuse_streams(scnir)
    settings = read_execution_settings({}, dt=dt)
    _vector64(sample)
    if type(model) is not nn.Sequential or not len(model):
        raise TypeError("Expected a nonempty flat torch.nn.Sequential")
    _refuse_streams(getattr(model, "scnir", None))
    _refuse_metadata(getattr(model, "nir_metadata", {}))
    if model._forward_hooks or model._forward_pre_hooks:
        raise ValueError("Forward hooks change sequence semantics and cannot be exported")
    modules = list(model.children())
    if len(modules) != len(model):
        raise ValueError("Shared module instances cannot be represented as independent NIR nodes")
    width = int(sample.numel())
    widths: list[int] = []
    modes: set[str] = set()
    for module in modules:
        if module._forward_hooks or module._forward_pre_hooks:
            raise ValueError("Forward hooks change module semantics and cannot be exported")
        if type(module) is nn.Linear:
            if module.in_features != width:
                raise ValueError("Linear input width disagrees with its predecessor")
            for value in module.parameters():
                _vector64(value.reshape(-1))
            width = int(module.out_features)
        elif type(module) in (snn.Leaky, snn.Synaptic):
            if module.reset_delay or module.inhibition or module.state_quant:
                raise ValueError("Delayed reset, inhibition and state quantisation are unsupported")
            if not module.init_hidden or module.output:
                raise ValueError("Cells must use init_hidden=True and output=False")
            if module.reset_mechanism not in ("zero", "subtract"):
                raise ValueError("Only immediate zero or subtract reset can be preserved")
            # These code identities are qualified against pinned snnTorch 1.0.0;
            # upgrades must pass the real installed-profile inference tests.
            if getattr(module.spike_grad, "__code__", None) not in (
                surrogate.atan().__code__,
                snn.Leaky._surrogate_bypass.__code__,
            ):
                raise ValueError(
                    "Only the standard ATan or disabled surrogate spike decision is qualified"
                )
            modes.add("reset" if module.reset_mechanism == "zero" else "snntorch-subtract")
            if not np.all(_parameter(module.graded_spikes_factor, width, "spike scale") == 1):
                raise ValueError("Graded spikes cannot be represented as binary NIR spikes")
            beta = _parameter(module.beta, width, "beta")
            if np.any(beta < 0) or np.any(beta > 1) or np.any(beta == 1) and not np.all(beta == 1):
                raise ValueError("beta must lie in [0,1), or be uniformly one for IF")
            if np.any(_parameter(module.threshold, width, "threshold") <= 0):
                raise ValueError("threshold must be positive")
            if type(module) is snn.Synaptic:
                alpha = _parameter(module.alpha, width, "alpha")
                if np.any(alpha < 0) or np.any(alpha >= 1) or np.any(beta == 1):
                    raise ValueError("Synaptic alpha and beta must lie in [0,1)")
        else:
            raise ValueError(f"Unsupported snnTorch layer {type(module).__name__}")
        widths.append(width)
    if len(modes) > 1:
        raise ValueError("Mixed reset conventions cannot be preserved by a single-clock SCNetwork")

    copied = deepcopy(model)
    copied_modules: list[Any] = list(copied.children())
    with torch.no_grad():
        for copied_cell in copied_modules:
            if type(copied_cell) in (snn.Leaky, snn.Synaptic):
                copied_cell.reset_mem()
        try:
            copied(sample.clone())
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(
                "The source sequence cannot execute the declared input and precision"
            ) from exc
    by_id = {id(module): size for module, size in zip(copied.children(), widths, strict=True)}

    def map_cell(module: Any) -> Any:
        size = by_id[id(module)]
        beta = _parameter(module.beta, size, "beta")
        threshold = _parameter(module.threshold, size, "threshold")
        zeros = np.zeros(size)
        if np.all(beta == 1):
            return nir.IF(r=np.full(size, 1 / settings.dt), v_threshold=threshold, v_reset=zeros)
        tau_mem = settings.dt / (1 - beta)
        if type(module) is snn.Leaky:
            return nir.LIF(tau_mem, tau_mem / settings.dt, zeros, threshold, zeros)
        alpha = _parameter(module.alpha, size, "alpha")
        tau_syn = settings.dt / (1 - alpha)
        return nir.CubaLIF(
            tau_syn=tau_syn,
            tau_mem=tau_mem,
            r=tau_mem / settings.dt,
            v_leak=zeros,
            v_threshold=threshold,
            w_in=tau_syn / settings.dt,
            v_reset=zeros,
        )

    graph = nirtorch.torch_to_nir(copied, {snn.Leaky: map_cell, snn.Synaptic: map_cell})
    graph.metadata = write_execution_metadata(
        graph.metadata, dt=settings.dt, reset_mode=next(iter(modes), "reset")
    )
    return from_nir(graph)


class _ScaledCell(_ModuleBase):
    """Apply the NIR input gain before a real snnTorch spiking cell."""

    gain: Any

    def __init__(self, cell: Any, gain: np.ndarray[Any, Any]) -> None:
        super().__init__()
        self.cell = cell
        self.register_buffer("gain", torch.from_numpy(gain.copy()))

    def forward(self, x: Any) -> Any:
        """Advance the real cell and return binary spikes in the input precision."""
        _vector64(x)
        # snnTorch's spike function returns float32 even for double membrane
        # state. Binary spikes convert exactly before the next double layer.
        return self.cell(x * self.gain).to(dtype=x.dtype)


def to_snntorch(network: SCNetwork, *, scnir: object | None = None) -> Any:
    """Export a supported SCNetwork chain as real Torch and snnTorch modules.

    Parameters
    ----------
    network : SCNetwork
        Single-input, single-output chain of Affine, Linear, IF, LIF and CubaLIF
        primitives. Leak and hard-reset potentials must be zero; decay factors
        must lie in [0,1]. Export starts from reset state, not current live state.
    scnir : object or None
        SC-NIR bitstream contracts cannot be preserved by float inference.

    Returns
    -------
    torch.nn.Sequential
        Independent CPU float64 layers with real snnTorch cells. Each call
        advances one timestep. Reset individual cells with their reset_mem().

    Raises
    ------
    TypeError
        The argument is not an SCNetwork floating reference graph.
    ValueError
        A primitive, topology, dynamics or stochastic contract is unsupported.
    """
    _refuse_streams(scnir)
    if not isinstance(network, SCNetwork):
        raise TypeError(
            "Expected an SCNetwork float graph; SC bitstream encoding and Q-format are unsupported"
        )
    if network.reset_mode == "subtract":
        raise ValueError(
            "SC subtract reset and snnTorch subtract reset differ after overshoot; "
            "use a graph imported with its explicit snntorch-subtract convention"
        )
    _refuse_metadata(network.nir_metadata)
    _refuse_metadata(network.nir_node_metadata)
    if any(note.kind == "approximated" for note in network.interchange_notes):
        raise ValueError(
            "Approximated NIR imports cannot be exported as faithful snnTorch inference"
        )
    graph = to_nir(network)
    names, edges = graph_order(graph)
    inputs = [name for name in names if isinstance(graph.nodes[name], nir.Input)]
    outputs = [name for name in names if isinstance(graph.nodes[name], nir.Output)]
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("snnTorch sequence export requires one input and one output")
    for name, attribute in [(inputs[0], "input_type"), (outputs[0], "output_type")]:
        ports = getattr(graph.nodes[name], attribute)
        if len(ports) != 1 or np.asarray(next(iter(ports.values()))).shape != (1,):
            raise ValueError("snnTorch interchange requires one vector at each graph boundary")
    chain, current = [], inputs[0]
    while current != outputs[0]:
        successors = [dst for src, dst in edges if src == current]
        if len(successors) != 1 or successors[0] in chain:
            raise ValueError(
                "Branches, recurrence and disconnected nodes require a different Torch executor"
            )
        current = successors[0]
        if len([src for src, dst in edges if dst == current]) != 1:
            raise ValueError("Merging edges cannot be represented as a Torch sequence")
        chain.append(current)
    if set(chain) | {inputs[0]} != set(names):
        raise ValueError("Disconnected NIR nodes cannot be discarded on export")
    layers: list[Any] = []
    for name in chain[:-1]:
        node = graph.nodes[name]
        if any(
            np.iscomplexobj(value) for value in vars(node).values() if isinstance(value, np.ndarray)
        ):
            raise ValueError(
                "Complex NIR parameters cannot be preserved by real-valued snnTorch cells"
            )
        if isinstance(node, (nir.Affine, nir.Linear)):
            weight = np.asarray(node.weight, dtype=np.float64)
            if (
                not np.isfinite(weight).all()
                or isinstance(node, nir.Affine)
                and not np.isfinite(node.bias).all()
            ):
                raise ValueError("snnTorch linear parameters must be finite")
            layer = nn.Linear(
                weight.shape[1],
                weight.shape[0],
                bias=isinstance(node, nir.Affine),
                dtype=torch.float64,
            )
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(weight.copy()))
                if isinstance(node, nir.Affine):
                    layer.bias.copy_(
                        torch.from_numpy(np.asarray(node.bias, dtype=np.float64).copy())
                    )
            layers.append(layer)
            continue
        if not isinstance(node, (nir.IF, nir.LIF, nir.CubaLIF)):
            raise ValueError(f"Unsupported snnTorch export primitive {type(node).__name__}")
        values = [
            np.asarray(getattr(node, key), dtype=np.float64).reshape(-1)
            for key in (
                ("r", "v_threshold", "v_reset")
                if isinstance(node, nir.IF)
                else ("r", "v_threshold", "v_reset", "v_leak")
            )
        ]
        if any(not np.isfinite(value).all() for value in values):
            raise ValueError("snnTorch export parameters must be finite")
        if np.any(node.v_reset != 0) or not isinstance(node, nir.IF) and np.any(node.v_leak != 0):
            raise ValueError(
                "Nonzero leak or reset potentials are not supported by this snnTorch profile"
            )
        threshold = np.asarray(node.v_threshold, dtype=np.float64).reshape(-1)
        if np.any(threshold <= 0):
            raise ValueError("threshold must be positive")
        tau = (
            None
            if isinstance(node, nir.IF)
            else node.tau_mem
            if isinstance(node, nir.CubaLIF)
            else node.tau
        )
        if tau is not None and (not np.isfinite(tau).all() or np.any(tau <= 0)):
            raise ValueError("NIR membrane time constants must be finite and positive")
        decay = (
            np.ones_like(threshold)
            if tau is None
            else 1 - network.dt / np.asarray(tau, dtype=np.float64).reshape(-1)
        )
        if not np.isfinite(decay).all() or np.any(decay < 0) or np.any(decay > 1):
            raise ValueError("NIR membrane decay cannot be preserved by snnTorch clamping")
        gain = np.asarray(node.r, dtype=np.float64).reshape(-1) * (
            network.dt if tau is None else 1 - decay
        )
        options = dict(
            beta=torch.from_numpy(decay.copy()),
            threshold=torch.from_numpy(threshold.copy()),
            init_hidden=True,
            reset_delay=False,
            surrogate_disable=True,
            reset_mechanism="zero" if network.reset_mode == "reset" else "subtract",
        )
        if isinstance(node, nir.CubaLIF):
            if not np.isfinite(node.tau_syn).all() or np.any(node.tau_syn <= 0):
                raise ValueError("NIR synaptic time constants must be finite and positive")
            alpha = 1 - network.dt / np.asarray(node.tau_syn, dtype=np.float64).reshape(-1)
            if not np.isfinite(alpha).all() or np.any(alpha < 0) or np.any(alpha >= 1):
                raise ValueError("NIR synaptic decay cannot be preserved by snnTorch clamping")
            gain = gain * np.asarray(node.w_in).reshape(-1) * (1 - alpha)
            cell = snn.Synaptic(alpha=torch.from_numpy(alpha.copy()), **options)
        else:
            cell = snn.Leaky(**options)
        if not np.isfinite(gain).all():
            raise ValueError("NIR input gain must be finite")
        layers.append(_ScaledCell(cell.double(), gain))
    return nn.Sequential(*layers).double()
