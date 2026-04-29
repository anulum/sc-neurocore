# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Generates Top-Level Verilog for a defined SC Network

import logging
from collections.abc import Mapping
from typing import Any, Dict

from .aer_emitter import AEREmitter
from ._ident import sanitize_ident
from .kuramoto_emitter import KuramotoEmitter
from .lfsr16_emitter import Lfsr16Emitter
from .sobol16_emitter import Sobol16Emitter

logger = logging.getLogger(__name__)

_SOURCE_NODE_TYPES = {
    "stochastic_source",
    "stochasticsource",
    "sc_stochastic_source",
    "sc_source",
    "source",
}
_LFSR_SOURCE_TYPES = {"lfsr", "lfsr16", "lfsr_16", "lfsr16_source", "sc_lfsr16_source"}
_SOBOL_SOURCE_TYPES = {"sobol", "sobol16", "sobol_16", "sobol16_source", "sc_sobol16_source"}


class VerilogGenerator:
    """
    Generates Top-Level Verilog for a defined SC Network.
    """

    def __init__(self, module_name="sc_network_top") -> None:  # type: ignore[no-untyped-def]
        self.module_name = sanitize_ident(module_name, context="module name")
        self.layers = []  # type: ignore[var-annotated]
        self.wires = []  # type: ignore[var-annotated]
        self.instances = []  # type: ignore[var-annotated]

    def add_layer(self, layer_type: str, name: str, params: Dict[str, Any]) -> None:
        self.layers.append(
            {
                "type": layer_type,
                "name": sanitize_ident(name, context="layer name"),
                "params": params,
            }
        )

    def generate(self, mode: str = "sync") -> str:
        """
        Emits Verilog code.
        """
        if mode == "async_aer":
            emitter = AEREmitter(module_name=self.module_name)
            for layer in self.layers:
                emitter.add_layer(layer["type"], layer["name"], layer["params"])
            return emitter.generate()
        if mode != "sync":
            raise ValueError("mode must be 'sync' or 'async_aer'")

        code = f"module {self.module_name} (\n"
        code += "    input wire clk,\n"
        code += "    input wire rst_n,\n"
        # Determine I/O from first/last layer logic (simplified)
        code += "    input wire [7:0] input_bus,\n"
        code += "    output wire [7:0] output_bus\n"
        code += ");\n\n"

        code += "    // Internal Signals\n"
        # Generate wires for connections
        for i in range(len(self.layers) - 1):
            code += f"    wire [7:0] layer_{i}_to_{i + 1};\n"

        code += "\n"

        # Instantiate Layers
        for i, layer in enumerate(self.layers):
            l_type = layer["type"]
            l_name = layer["name"]

            # Simple Dense Layer instantiation logic
            if l_type == "Dense":
                code += f"    // Layer {i}: {l_name}\n"
                code += "    sc_dense_layer_core #(\n"
                code += f"        .NUM_NEURONS({layer['params'].get('n_neurons', 10)})\n"
                code += f"    ) {l_name}_inst (\n"
                code += "        .clk(clk),\n"
                code += "        .rst_n(rst_n),\n"

                # Connect Input
                if i == 0:
                    code += "        .input_bus(input_bus),\n"
                else:
                    code += f"        .input_bus(layer_{i - 1}_to_{i}),\n"

                # Connect Output
                if i == len(self.layers) - 1:
                    code += "        .output_bus(output_bus)\n"
                else:
                    code += f"        .output_bus(layer_{i}_to_{i + 1})\n"

                code += "    );\n\n"

        code += "endmodule\n"
        source_modules = emit_sources_from_ir({"nodes": self.layers})
        if source_modules:
            code += f"\n\n{source_modules}\n"
        return code

    def emit_lfsr16_source(self, module_name: str = "sc_lfsr16_source", seed: int = 0xACE1) -> str:
        """Emit a standalone LFSR-16 stochastic source module."""
        return Lfsr16Emitter(module_name=module_name, seed=seed).generate()

    def emit_sobol16_source(self, module_name: str = "sc_sobol16_source", seed: int = 0) -> str:
        """Emit a standalone Sobol-16 stochastic source module."""
        return Sobol16Emitter(module_name=module_name, seed=seed).generate()

    def emit_sources_from_ir(self, ir: Any) -> str:
        """Emit standalone stochastic source modules declared in an IR payload."""
        return emit_sources_from_ir(ir)

    def emit_async_aer(self, module_name: str | None = None) -> str:
        """Emit the research-stage async AER wrapper."""
        emitter = AEREmitter(module_name=module_name or self.module_name)
        for layer in self.layers:
            emitter.add_layer(layer["type"], layer["name"], layer["params"])
        return emitter.generate()

    def emit_kuramoto_phase(
        self,
        module_name: str | None = None,
        *,
        n_oscillators: int = 4,
        omegas: list[float] | tuple[float, ...] | None = None,
        initial_phases: list[float] | tuple[float, ...] | None = None,
        coupling: float = 0.1,
        dt: float = 1e-2,
        data_width: int = 24,
        fraction: int = 16,
        lut_size: int = 64,
    ) -> str:
        """Emit the bounded research Kuramoto phase core."""
        emitter = KuramotoEmitter(
            module_name=module_name or self.module_name,
            n_oscillators=n_oscillators,
            omegas=omegas,
            initial_phases=initial_phases,
            coupling=coupling,
            dt=dt,
            data_width=data_width,
            fraction=fraction,
            lut_size=lut_size,
        )
        return emitter.generate()

    def save_to_file(self, path: str) -> None:
        try:
            with open(path, "w") as f:
                f.write(self.generate())
        except OSError as exc:
            logger.error("Failed to write Verilog to %s: %s", path, exc)
            raise


def emit_sources_from_ir(ir: Any) -> str:
    """Emit LFSR-16 and Sobol-16 source modules from a lightweight IR payload.

    The helper accepts the mapping shapes already used by documentation,
    tests, and compiler-service payloads: ``{"nodes": [...]}``,
    ``{"nodes": {"node_id": {...}}}``, or a direct iterable of node mappings.
    Non-source nodes are ignored. Source nodes must identify their generator
    through ``source_type``, ``decorrelator``, ``generator``, ``strategy``, or
    the node ``type``/``node_type`` itself.
    """
    emitted = []
    seen_names: set[str] = set()
    for index, (node_id, node) in enumerate(_iter_ir_nodes(ir)):
        kind = _source_kind(node)
        if kind is None:
            continue
        module_name = _source_module_name(node, node_id=node_id, index=index)
        if module_name in seen_names:
            raise ValueError(f"duplicate stochastic source module name {module_name!r}")
        seen_names.add(module_name)
        seed = _source_seed(node, default=0xACE1 if kind == "lfsr16" else 0)
        if kind == "lfsr16":
            emitted.append(Lfsr16Emitter(module_name=module_name, seed=seed).generate())
        elif kind == "sobol16":
            emitted.append(Sobol16Emitter(module_name=module_name, seed=seed).generate())
        else:
            raise ValueError(f"unsupported stochastic source type {kind!r}")
    return "\n\n".join(emitted)


def _iter_ir_nodes(ir: Any) -> list[tuple[str | None, Any]]:
    if isinstance(ir, Mapping):
        nodes = ir.get("nodes", ir)
    else:
        nodes = getattr(ir, "nodes", ir)

    if isinstance(nodes, Mapping):
        return [(str(node_id), node) for node_id, node in nodes.items()]
    if isinstance(nodes, (list, tuple)):
        return [(None, node) for node in nodes]
    raise TypeError("IR payload must contain a mapping or sequence of nodes")


def _source_kind(node: Any) -> str | None:
    params = _node_params(node)
    node_type = _normalise(_node_value(node, "type", "node_type", "op", "kind"))
    candidate = _normalise(
        _node_value(
            node,
            "source_type",
            "decorrelator",
            "generator",
            "strategy",
            default=_node_value(params, "source_type", "decorrelator", "generator", "strategy"),
        )
    )

    if node_type in _LFSR_SOURCE_TYPES or candidate in _LFSR_SOURCE_TYPES:
        return "lfsr16"
    if node_type in _SOBOL_SOURCE_TYPES or candidate in _SOBOL_SOURCE_TYPES:
        return "sobol16"
    if node_type in _SOURCE_NODE_TYPES:
        if candidate:
            raise ValueError(f"unsupported stochastic source type {candidate!r}")
        raise ValueError("stochastic source node is missing source_type/decorrelator")
    return None


def _source_module_name(node: Any, *, node_id: str | None, index: int) -> str:
    params = _node_params(node)
    raw_name = _node_value(
        node,
        "module_name",
        "name",
        "id",
        "node_id",
        default=_node_value(params, "module_name", "name", "id", "node_id", default=node_id),
    )
    if raw_name is None:
        raw_name = f"sc_stochastic_source_{index}"
    return sanitize_ident(str(raw_name), context="stochastic source module name")


def _source_seed(node: Any, *, default: int) -> int:
    params = _node_params(node)
    raw_seed = _node_value(node, "seed", default=_node_value(params, "seed", default=default))
    return int(raw_seed)


def _node_params(node: Any) -> Any:
    return _node_value(node, "params", "parameters", "attrs", "attributes", default={})


def _node_value(node: Any, *keys: str, default: Any = None) -> Any:
    if isinstance(node, Mapping):
        for key in keys:
            if key in node:
                return node[key]
        return default
    for key in keys:
        if hasattr(node, key):
            return getattr(node, key)
    return default


def _normalise(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("-", "_")
