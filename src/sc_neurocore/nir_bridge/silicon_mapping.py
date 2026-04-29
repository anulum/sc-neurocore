# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR silicon mapping reports

"""Deterministic NIR silicon mapping reports for neuromorphic targets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from sc_neurocore.nir_bridge.hardware_targets import (
    build_noise_annotation,
    get_hardware_profile,
)

SCHEMA_VERSION = "sc-neurocore.nir-silicon-mapping.v1"
_DEFAULT_TARGETS = ("loihi2", "spinnaker2")

_NODE_TYPE_ALIASES = {
    "_UnitDelayNode": "Delay",
    "SCAffineNode": "Affine",
    "SCAvgPool2dNode": "AvgPool2d",
    "SCConv1dNode": "Conv1d",
    "SCConv2dNode": "Conv2d",
    "SCCubaLINode": "CubaLI",
    "SCCubaLIFNode": "CubaLIF",
    "SCDelayNode": "Delay",
    "SCFlattenNode": "Flatten",
    "SCIFNode": "IF",
    "SCInputNode": "Input",
    "SCIntegratorNode": "I",
    "SCLIFNode": "LIF",
    "SCLINode": "LI",
    "SCLinearNode": "Linear",
    "SCOutputNode": "Output",
    "SCScaleNode": "Scale",
    "SCSubgraphNode": "NIRGraph",
    "SCMultiPortSubgraphNode": "NIRGraph",
    "SCThresholdNode": "Threshold",
    "SCSumPool2dNode": "SumPool2d",
}


@dataclass(frozen=True)
class SiliconMappingConfig:
    """Configuration for NIR silicon mapping report generation."""

    targets: tuple[str, ...] = _DEFAULT_TARGETS
    bitstream_length: int = 256
    event_rate_hz: float = 1000.0
    noise_observations: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    artefact_name: str = "nir_silicon_mapping_report.json"

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("targets must not be empty")
        if self.bitstream_length <= 0:
            raise ValueError("bitstream_length must be positive")
        if self.event_rate_hz <= 0.0 or not math.isfinite(self.event_rate_hz):
            raise ValueError("event_rate_hz must be finite and positive")
        for target in self.targets:
            get_hardware_profile(target)


@dataclass(frozen=True)
class _GraphView:
    nodes: dict[str, Any]
    order: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]


def build_silicon_mapping_report(
    source: Any,
    config: SiliconMappingConfig | None = None,
) -> dict[str, Any]:
    """Build a deterministic target-mapping report for a parsed NIR network."""

    cfg = config or SiliconMappingConfig()
    graph = _coerce_graph(source)
    node_payloads = [_node_payload(name, graph.nodes[name]) for name in graph.order]

    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "topological_order": list(graph.order),
        },
        "targets": [
            _target_report(target, node_payloads, graph.edges, cfg) for target in cfg.targets
        ],
    }


def write_silicon_mapping_report(
    output_dir: str | Path,
    source: Any,
    config: SiliconMappingConfig | None = None,
) -> Path:
    """Write `nir_silicon_mapping_report.json` in deterministic form."""

    cfg = config or SiliconMappingConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / cfg.artefact_name
    path.write_text(
        json.dumps(build_silicon_mapping_report(source, cfg), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _coerce_graph(source: Any) -> _GraphView:
    if isinstance(source, Mapping):
        nodes = {str(name): node for name, node in source.items()}
        return _GraphView(nodes=nodes, order=tuple(sorted(nodes)), edges=())

    raw_nodes = getattr(source, "nodes", None)
    if not isinstance(raw_nodes, Mapping):
        raise TypeError("source must be a parsed NIR network or a mapping of node names to nodes")

    nodes = {str(name): node for name, node in raw_nodes.items()}
    raw_order = getattr(source, "topo_order", None)
    if raw_order is None:
        order = tuple(sorted(nodes))
    else:
        order = tuple(str(name) for name in raw_order if str(name) in nodes)
        missing = tuple(sorted(set(nodes) - set(order)))
        order = order + missing

    edges = tuple(
        (str(src), str(dst))
        for src, dst in getattr(source, "edges", ())
        if str(src) in nodes and str(dst) in nodes
    )
    return _GraphView(nodes=nodes, order=order, edges=edges)


def _node_payload(name: str, node: Any) -> dict[str, Any]:
    node_type = _node_type(node)
    resources = _resource_estimate(node, node_type)
    return {
        "name": name,
        "node_type": node_type,
        "resource_estimate": resources,
    }


def _node_type(node: Any) -> str:
    if isinstance(node, str):
        return node
    if isinstance(node, Mapping):
        value = node.get("node_type", node.get("type", "Unknown"))
        return str(value)
    raw = getattr(node, "nir_node_type", None)
    if raw is not None:
        return str(raw)
    return _NODE_TYPE_ALIASES.get(type(node).__name__, type(node).__name__)


def _resource_estimate(node: Any, node_type: str) -> dict[str, int]:
    weight = _mapping_or_attr(node, "weight")
    weight_shape = _shape(weight)
    weight_count = _shape_size(weight_shape)
    n_neurons = _int_attr(node, "n_neurons", 0)

    if weight_shape:
        n_neurons = max(n_neurons, int(weight_shape[0]))
    elif node_type in {"Input", "Output"}:
        n_neurons = _shape_size(_shape(_mapping_or_attr(node, "shape")))
    elif node_type in {"SumPool2d", "AvgPool2d"}:
        kernel = _mapping_or_attr(node, "kernel_size")
        weight_count = _shape_product(kernel)
        n_neurons = 1
    elif node_type in {"Flatten", "Delay"}:
        n_neurons = 1 if n_neurons == 0 else n_neurons

    return {
        "abstract_neurons": max(0, n_neurons),
        "abstract_synapses": max(0, weight_count),
        "state_bytes_estimate": max(0, n_neurons) * 8,
        "weight_bytes_estimate": max(0, weight_count) * 2,
    }


def _target_report(
    target_id: str,
    node_payloads: list[dict[str, Any]],
    edges: tuple[tuple[str, str], ...],
    config: SiliconMappingConfig,
) -> dict[str, Any]:
    profile = get_hardware_profile(target_id)
    supported = set(profile.supported_nir_nodes)
    unsupported = set(profile.unsupported_nir_nodes)
    native_bitstream = config.bitstream_length in profile.sc_constraints.bitstream_lengths

    mapped_nodes = []
    for node in node_payloads:
        lowering, diagnostics = _lowering_result(
            node["node_type"], supported, unsupported, native_bitstream
        )
        mapped_nodes.append(
            {
                **node,
                "lowering": lowering,
                "diagnostics": diagnostics,
            }
        )

    summary = _summary(mapped_nodes, edges, native_bitstream)
    status = _status(summary)
    hooks = _noise_hooks(profile.sc_constraints.back_annotation_channels)
    payload = {
        "backend_status": profile.backend_status,
        "display_name": profile.display_name,
        "target_id": profile.target_id,
        "lowering_status": status,
        "summary": {
            **summary,
            "event_rate_hz": config.event_rate_hz,
            "selected_bitstream_length": config.bitstream_length,
        },
        "nodes": mapped_nodes,
        "fallback_requirements": _fallback_requirements(mapped_nodes, native_bitstream),
        "noise_back_annotation_hooks": hooks,
        "limitations": [
            "report is a deterministic planning artefact; no vendor SDK or hardware is invoked",
            "resource estimates are abstract graph estimates, not placement-legal hardware utilisation",
            "hardware-noise replay requires measured observations from the target platform",
        ],
    }

    observations = config.noise_observations.get(profile.target_id)
    if observations is not None:
        payload["noise_annotation"] = build_noise_annotation(
            profile.target_id, observations
        ).to_dict()
    return payload


def _lowering_result(
    node_type: str,
    supported: set[str],
    unsupported: set[str],
    native_bitstream: bool,
) -> tuple[str, list[str]]:
    diagnostics: list[str] = []
    if node_type in supported:
        lowering = "native"
        diagnostics.append("node supported by target manifest")
    elif node_type in unsupported:
        lowering = "fallback"
        diagnostics.append("node requires host-side or pre-lowering fallback")
    else:
        lowering = "unsupported"
        diagnostics.append("node type is not listed in the target manifest")

    if not native_bitstream:
        diagnostics.append("bitstream length requires resampling for this target")
    return lowering, diagnostics


def _summary(
    mapped_nodes: list[dict[str, Any]],
    edges: tuple[tuple[str, str], ...],
    native_bitstream: bool,
) -> dict[str, Any]:
    resources = [node["resource_estimate"] for node in mapped_nodes]
    return {
        "native_nodes": sum(1 for node in mapped_nodes if node["lowering"] == "native"),
        "fallback_nodes": sum(1 for node in mapped_nodes if node["lowering"] == "fallback"),
        "unsupported_nodes": sum(1 for node in mapped_nodes if node["lowering"] == "unsupported"),
        "native_bitstream_length": native_bitstream,
        "estimated_neurons": sum(int(item["abstract_neurons"]) for item in resources),
        "estimated_synapses": sum(int(item["abstract_synapses"]) for item in resources),
        "estimated_state_bytes": sum(int(item["state_bytes_estimate"]) for item in resources),
        "estimated_weight_bytes": sum(int(item["weight_bytes_estimate"]) for item in resources),
        "routing_edges": len(edges),
    }


def _status(summary: Mapping[str, Any]) -> str:
    if int(summary["unsupported_nodes"]) > 0:
        return "unsupported"
    if int(summary["fallback_nodes"]) > 0 or not bool(summary["native_bitstream_length"]):
        return "fallback_required"
    return "clean"


def _fallback_requirements(
    mapped_nodes: list[dict[str, Any]], native_bitstream: bool
) -> list[dict[str, str]]:
    requirements = [
        {
            "node": node["name"],
            "node_type": node["node_type"],
            "requirement": "pre-lower or host-side execute before silicon mapping",
        }
        for node in mapped_nodes
        if node["lowering"] in {"fallback", "unsupported"}
    ]
    if not native_bitstream:
        requirements.append(
            {
                "node": "*",
                "node_type": "SCBitstream",
                "requirement": "resample stochastic bitstreams to a target-supported length",
            }
        )
    return requirements


def _noise_hooks(channels: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            "channel": channel,
            "requires_measured_hardware": True,
            "simulation_replay": "deterministic_seeded",
        }
        for channel in channels
    ]


def _mapping_or_attr(node: Any, key: str) -> Any:
    if isinstance(node, Mapping):
        return node.get(key)
    return getattr(node, key, None)


def _int_attr(node: Any, key: str, default: int) -> int:
    value = _mapping_or_attr(node, key)
    if value is None:
        return default
    return int(value)


def _shape(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return tuple(int(item) for item in value)
    if isinstance(value, list):
        return tuple(int(item) for item in value)
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(item) for item in shape)
    return ()


def _shape_size(shape: tuple[int, ...]) -> int:
    if not shape:
        return 0
    total = 1
    for item in shape:
        total *= max(1, int(item))
    return total


def _shape_product(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, tuple | list):
        total = 1
        for item in value:
            total *= int(item)
        return total
    return 0
