# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Generates Top-Level Verilog for a defined SC Network

import logging
from collections.abc import Mapping
from numbers import Integral
from typing import Any, Dict

from .aer_emitter import AEREmitter
from ._ident import sanitize_ident
from .kuramoto_emitter import KuramotoEmitter
from .lfsr16_emitter import Lfsr16Emitter
from .sobol16_emitter import Sobol16Emitter
from .quasirandom_emitter import QuasiRandomEmitter, Halton16Emitter

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
_HALTON_SOURCE_TYPES = {"halton", "halton16", "halton_16", "halton16_source", "sc_halton16_source"}
_SYNC_AUXILIARY_LAYER_TYPES = {"StochasticSource"}


class VerilogGenerator:
    """
    Generates Top-Level Verilog for a defined SC Network.
    """

    def __init__(self, module_name: str = "sc_network_top", bus_width: int = 8) -> None:
        """Initialise with a top-level module name."""
        self.module_name = sanitize_ident(module_name, context="module name")
        self.bus_width = self._require_positive_int(bus_width, "bus_width")
        self.layers = []  # type: ignore[var-annotated]
        self.wires = []  # type: ignore[var-annotated]
        self.instances = []  # type: ignore[var-annotated]

    def add_layer(self, layer_type: str, name: str, params: Dict[str, Any]) -> None:
        """Add a layer definition to the network."""
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
            emitter = AEREmitter(module_name=self.module_name, bus_width=self.bus_width)
            for layer in self.layers:
                emitter.add_layer(layer["type"], layer["name"], layer["params"])
            return emitter.generate()
        if mode != "sync":
            raise ValueError("mode must be 'sync' or 'async_aer'")
        self._validate_sync_layers()
        layer_widths = self._sync_layer_widths()
        input_width = layer_widths[0][0] if layer_widths else self.bus_width
        output_width = layer_widths[-1][1] if layer_widths else self.bus_width

        code = f"module {self.module_name} (\n"
        code += "    input wire clk,\n"
        code += "    input wire rst_n,\n"
        code += f"    input wire [{input_width - 1}:0] input_bus,\n"
        code += f"    output wire [{output_width - 1}:0] output_bus\n"
        code += ");\n\n"

        code += "    // Internal Signals\n"
        # Generate wires for connections
        for i in range(len(layer_widths) - 1):
            code += f"    wire [{layer_widths[i][1] - 1}:0] layer_{i}_to_{i + 1};\n"

        code += "\n"

        # Instantiate Layers
        dense_idx = 0
        for i, layer in enumerate(self.layers):
            l_type = layer["type"]
            l_name = layer["name"]

            if l_type == "Dense":
                code += f"    // Layer {i}: {l_name}\n"
                code += "    sc_dense_layer_core #(\n"
                code += f"        .NUM_NEURONS({layer['params']['n_neurons']})\n"
                code += f"    ) {l_name}_inst (\n"
                code += "        .clk(clk),\n"
                code += "        .rst_n(rst_n),\n"

                # Connect Input
                if dense_idx == 0:
                    code += "        .input_bus(input_bus),\n"
                else:
                    code += f"        .input_bus(layer_{dense_idx - 1}_to_{dense_idx}),\n"

                # Connect Output
                if dense_idx == len(layer_widths) - 1:
                    code += "        .output_bus(output_bus)\n"
                else:
                    code += f"        .output_bus(layer_{dense_idx}_to_{dense_idx + 1})\n"

                code += "    );\n\n"
                dense_idx += 1

        code += "endmodule\n"
        source_modules = emit_sources_from_ir({"nodes": self.layers})
        if source_modules:
            code += f"\n\n{source_modules}\n"
        return code

    def _validate_sync_layers(self) -> None:
        """Reject sync RTL configurations that cannot be emitted faithfully."""
        for layer in self.layers:
            layer_type = layer["type"]
            layer_name = layer["name"]
            params = layer["params"]
            if layer_type == "Dense":
                if "n_neurons" not in params:
                    raise ValueError(f"Dense layer '{layer_name}' requires n_neurons")
                self._require_positive_int(
                    params["n_neurons"],
                    f"Dense layer '{layer_name}' n_neurons",
                )
                for width_name in ("input_width", "output_width"):
                    if width_name in params:
                        self._require_positive_int(
                            params[width_name],
                            f"Dense layer '{layer_name}' {width_name}",
                        )
                continue
            if layer_type in _SYNC_AUXILIARY_LAYER_TYPES:
                continue
            raise ValueError(f"unsupported sync layer type '{layer_type}' for layer '{layer_name}'")

    @staticmethod
    def _require_positive_int(value: Any, name: str) -> int:
        """Return value as int after rejecting booleans and non-positive values."""
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    def _dense_input_width(self, params: Mapping[str, Any], previous_width: int | None) -> int:
        if "input_width" in params:
            return self._require_positive_int(params["input_width"], "input_width")
        return previous_width if previous_width is not None else self.bus_width

    def _dense_output_width(self, params: Mapping[str, Any]) -> int:
        if "output_width" in params:
            return self._require_positive_int(params["output_width"], "output_width")
        return self._require_positive_int(params["n_neurons"], "n_neurons")

    def _sync_layer_widths(self) -> list[tuple[int, int]]:
        """Return per-layer ``(input_width, output_width)`` and reject mismatches."""
        widths: list[tuple[int, int]] = []
        previous_width: int | None = None
        previous_name: str | None = None
        for layer in self.layers:
            if layer["type"] != "Dense":
                continue
            name = layer["name"]
            params = layer["params"]
            input_width = self._dense_input_width(params, previous_width)
            output_width = self._dense_output_width(params)
            if previous_width is not None and input_width != previous_width:
                raise ValueError(
                    f"{previous_name} -> {name} width mismatch: "
                    f"{previous_width} output bits cannot drive {input_width} input bits"
                )
            widths.append((input_width, output_width))
            previous_width = output_width
            previous_name = name
        return widths

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

    def emit_halton16_source(self, module_name: str = "sc_halton16_source") -> str:
        """Emit a standalone Halton-16 stochastic source module."""
        return Halton16Emitter(module_name=module_name).generate()

    def emit_quasirandom_source(
        self,
        method: str = "sobol",
        module_name: str | None = None,
        seed: int = 0,
    ) -> str:
        """Emit a quasi-random source via the unified factory.

        Parameters
        ----------
        method : str
            ``"sobol"`` or ``"halton"``.
        module_name : str, optional
            Override the default module name.
        seed : int
            Seed for Sobol (ignored for Halton).
        """
        if method not in {"sobol", "halton"}:
            raise ValueError("method must be 'sobol' or 'halton'")

        if method == "sobol":
            return QuasiRandomEmitter(
                method="sobol",
                module_name=module_name,
                seed=seed,
            ).generate()
        return QuasiRandomEmitter(
            method="halton",
            module_name=module_name,
            seed=seed,
        ).generate()

    def emit_decorrelator(
        self,
        *,
        num_streams: int = 8,
        stream_width: int = 16,
        shift_seed: int = 0xA5A5_5A5A,
    ) -> str:
        """Return the path to the sc_decorrelator HDL module.

        The decorrelator is a static Verilog module — this method provides
        the instantiation template for integration into top-level designs.
        """
        return (
            f"    sc_decorrelator #(\n"
            f"        .NUM_STREAMS({num_streams}),\n"
            f"        .STREAM_WIDTH({stream_width}),\n"
            f"        .SHIFT_SEED(32'h{shift_seed:08X})\n"
            f"    ) decorrelator_inst (\n"
            f"        .clk(clk),\n"
            f"        .rst_n(rst_n),\n"
            f"        .source_bits(source_bits),\n"
            f"        .decorrelated(decorrelated_bus)\n"
            f"    );\n"
        )

    def emit_edt_controller(
        self,
        *,
        data_width: int = 16,
        margin: int = 0x0040,
        stable_cycles: int = 8,
    ) -> str:
        """Return an instantiation template for the EDT controller."""
        return (
            f"    sc_edt_controller #(\n"
            f"        .DATA_WIDTH({data_width}),\n"
            f"        .MARGIN(16'h{margin:04X}),\n"
            f"        .STABLE_CYCLES({stable_cycles})\n"
            f"    ) edt_inst (\n"
            f"        .clk(clk),\n"
            f"        .rst_n(rst_n),\n"
            f"        .enable(edt_enable),\n"
            f"        .accumulator(accumulator),\n"
            f"        .threshold(threshold),\n"
            f"        .decision_ready(decision_ready),\n"
            f"        .decision_value(decision_value),\n"
            f"        .freeze(freeze)\n"
            f"    );\n"
        )

    def emit_tmr_wrapper(
        self,
        module_name: str,
        inputs: list[tuple[str, int]],
        outputs: list[tuple[str, int]],
    ) -> str:
        """Generate a TMR wrapper for the given module."""
        from .tmr_wrapper import generate_tmr_wrapper

        return generate_tmr_wrapper(module_name=module_name, inputs=inputs, outputs=outputs)

    def save_to_file(self, path: str) -> None:
        """Write generated Verilog to a file."""
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
            # _source_kind() only ever returns lfsr16/sobol16/halton16 (None is
            # filtered above; an unknown candidate raises inside it), so the
            # remaining case is always halton16.
            emitted.append(Halton16Emitter(module_name=module_name).generate())
    return "\n\n".join(emitted)


def _iter_ir_nodes(ir: Any) -> list[tuple[str | None, Any]]:
    """Iterate IR nodes, yielding (node_id, node) pairs."""
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
    """Determine the stochastic source kind from an IR node."""
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
    if node_type in _HALTON_SOURCE_TYPES or candidate in _HALTON_SOURCE_TYPES:
        return "halton16"
    if node_type in _SOURCE_NODE_TYPES:
        if candidate:
            raise ValueError(f"unsupported stochastic source type {candidate!r}")
        raise ValueError("stochastic source node is missing source_type/decorrelator")
    return None


def _source_module_name(node: Any, *, node_id: str | None, index: int) -> str:
    """Extract or generate a unique module name for a stochastic source."""
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
    """Extract the PRNG seed from an IR node."""
    params = _node_params(node)
    raw_seed = _node_value(node, "seed", default=_node_value(params, "seed", default=default))
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, Integral):
        raise ValueError("stochastic source seed must be an integer")
    return int(raw_seed)


def _node_params(node: Any) -> Any:
    """Extract the params/attributes sub-mapping from an IR node."""
    return _node_value(node, "params", "parameters", "attrs", "attributes", default={})


def _node_value(node: Any, *keys: str, default: Any = None) -> Any:
    """Look up the first matching key in a node mapping or object."""
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
    """Normalise a node type string to lowercase with underscores."""
    if value is None:
        return ""
    return str(value).strip().lower().replace("-", "_")
