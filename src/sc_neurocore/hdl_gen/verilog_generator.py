# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Top-level Verilog generator for stochastic neural networks

"""Generate top-level Verilog wrappers and stochastic-source modules."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from numbers import Integral
from typing import Any

from ._ident import sanitize_ident
from .aer_emitter import (
    AEREmitter,
    compute_dense_layer_widths,
    render_dense_layer_instance,
    require_positive_int,
    validate_dense_layer_params,
)
from .kuramoto_emitter import KuramotoEmitter
from .lfsr16_emitter import Lfsr16Emitter
from .quasirandom_emitter import Halton16Emitter, QuasiRandomEmitter
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
_HALTON_SOURCE_TYPES = {"halton", "halton16", "halton_16", "halton16_source", "sc_halton16_source"}
_SYNC_AUXILIARY_LAYER_TYPES = {"StochasticSource"}


class VerilogGenerator:
    """Generate top-level Verilog for a declared stochastic-computing network."""

    def __init__(self, module_name: str = "sc_network_top", bus_width: int = 8) -> None:
        """Initialise the top-level Verilog generator.

        Parameters
        ----------
        module_name : str
            Sanitised top-level Verilog module identifier.
        bus_width : int
            Default input/output bitstream bus width in bits.
        """
        self.module_name = sanitize_ident(module_name, context="module name")
        self.bus_width = self._require_positive_int(bus_width, "bus_width")
        self.layers: list[dict[str, Any]] = []
        self.wires: list[str] = []
        self.instances: list[str] = []

    def add_layer(self, layer_type: str, name: str, params: dict[str, Any]) -> None:
        """Append a network layer specification.

        Parameters
        ----------
        layer_type : str
            Layer kind (``"Dense"`` or ``"StochasticSource"``).
        name : str
            Sanitised layer identifier.
        params : dict[str, Any]
            Layer parameter dictionary. For ``"Dense"``, ``n_neurons`` sets the output width,
            and ``weights`` (or one ``weight_fp`` for every input) gives one unsigned
            probability word per input. ``sc_dense_layer_core`` holds one weight per input
            and one dot product, so every neuron of the layer receives the same current
            with the same leak and gain: its ``n_neurons`` output bits always carry the
            same value. Per-neuron weights need a different core.
        """
        self.layers.append(
            {
                "type": layer_type,
                "name": sanitize_ident(name, context="layer name"),
                "params": params,
            }
        )

    def generate(self, mode: str = "sync") -> str:
        """Emit complete Verilog source in synchronous or clocked-AER mode.

        Parameters
        ----------
        mode : str
            ``"sync"`` or ``"async_aer"``.

        Returns
        -------
        str
            Generated Verilog module text.
        """
        if mode == "async_aer":
            return self.emit_async_aer()
        if mode != "sync":
            raise ValueError("mode must be 'sync' or 'async_aer'")
        self._validate_sync_layers()
        layer_widths = compute_dense_layer_widths(self.layers, self.bus_width)
        input_width = layer_widths[0][0] if layer_widths else self.bus_width
        output_width = layer_widths[-1][1] if layer_widths else self.bus_width

        lines = [
            f"module {self.module_name} (",
            "    input wire clk,",
            "    input wire rst_n,",
            f"    input wire [{input_width - 1}:0] input_bus,",
            f"    output wire [{output_width - 1}:0] output_bus",
            ");",
            "",
            "    // Internal Signals",
        ]
        for i in range(len(layer_widths) - 1):
            lines.append(f"    wire [{layer_widths[i][1] - 1}:0] layer_{i}_to_{i + 1};")
        if layer_widths:
            lines.append("    genvar sc_in_idx;")
        lines.append("")

        dense_idx = 0
        for i, layer in enumerate(self.layers):
            if layer["type"] != "Dense":
                continue
            in_sig = "input_bus" if dense_idx == 0 else f"layer_{dense_idx - 1}_to_{dense_idx}"
            out_sig = (
                "output_bus"
                if dense_idx == len(layer_widths) - 1
                else f"layer_{dense_idx}_to_{dense_idx + 1}"
            )
            lines.append(
                render_dense_layer_instance(
                    comment_label=f"Layer {i}: {layer['name']}",
                    layer_name=str(layer["name"]),
                    params=layer["params"],
                    input_bus=in_sig,
                    output_bus=out_sig,
                    input_width=layer_widths[dense_idx][0],
                )
            )
            dense_idx += 1

        if not layer_widths:
            lines.extend(
                [
                    f"    assign output_bus = {output_width}'b0;",
                    "    wire _unused_empty_ports = &{1'b0, clk, rst_n, input_bus, 1'b0};",
                    "",
                ]
            )

        lines.append("endmodule")
        code = "\n".join(lines) + "\n"
        source_modules = emit_sources_from_ir({"nodes": self.layers})
        if source_modules:
            code += f"\n\n{source_modules}\n"
        return code

    def _validate_sync_layers(self) -> None:
        """Reject sync RTL configurations that cannot be emitted faithfully."""
        for layer in self.layers:
            layer_type = layer["type"]
            layer_name = str(layer["name"])
            if layer_type == "Dense":
                validate_dense_layer_params(layer_name, layer["params"])
                continue
            if layer_type in _SYNC_AUXILIARY_LAYER_TYPES:
                continue
            raise ValueError(f"unsupported sync layer type '{layer_type}' for layer '{layer_name}'")

    @staticmethod
    def _require_positive_int(value: Any, name: str) -> int:
        """Return ``value`` as ``int`` after rejecting booleans and non-positive values."""
        return require_positive_int(value, name)

    def emit_lfsr16_source(self, module_name: str = "sc_lfsr16_source", seed: int = 0xACE1) -> str:
        """Emit a standalone LFSR-16 stochastic source module.

        Parameters
        ----------
        module_name : str
            Target Verilog module identifier.
        seed : int
            Initial 16-bit LFSR state.

        Returns
        -------
        str
            Generated Verilog module text.
        """
        return Lfsr16Emitter(module_name=module_name, seed=seed).generate()

    def emit_sobol16_source(self, module_name: str = "sc_sobol16_source", seed: int = 0) -> str:
        """Emit a standalone Sobol-16 stochastic source module.

        Parameters
        ----------
        module_name : str
            Target Verilog module identifier.
        seed : int
            Initial 16-bit scramble seed.

        Returns
        -------
        str
            Generated Verilog module text.
        """
        return Sobol16Emitter(module_name=module_name, seed=seed).generate()

    def emit_sources_from_ir(self, ir: Any) -> str:
        """Emit standalone stochastic source modules declared in an IR payload.

        Parameters
        ----------
        ir : Any
            Mapping, object with ``nodes``, or sequence of IR node entries.

        Returns
        -------
        str
            Concatenated Verilog source modules.
        """
        return emit_sources_from_ir(ir)

    def emit_async_aer(self, module_name: str | None = None) -> str:
        """Emit the clocked AER wrapper around the declared ``Dense`` layers.

        Parameters
        ----------
        module_name : str or None
            Optional override for the top-level module identifier.

        Returns
        -------
        str
            Generated Verilog module text.
        """
        emitter = AEREmitter(
            module_name=module_name or self.module_name,
            bus_width=self.bus_width,
        )
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
        """Emit the bounded research Kuramoto phase core.

        Parameters
        ----------
        module_name : str or None
            Optional override for the module identifier.
        n_oscillators : int
            Number of coupled phase oscillators.
        omegas : list[float] or tuple[float, ...] or None
            Natural frequencies per oscillator.
        initial_phases : list[float] or tuple[float, ...] or None
            Initial phase angles in radians.
        coupling : float
            Global coupling strength.
        dt : float
            Euler integration step size.
        data_width : int
            Fixed-point word width in bits.
        fraction : int
            Fractional bit count.
        lut_size : int
            Sine lookup-table depth.

        Returns
        -------
        str
            Generated Verilog module text.
        """
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
        """Emit a standalone Halton-16 stochastic source module.

        Parameters
        ----------
        module_name : str
            Target Verilog module identifier.

        Returns
        -------
        str
            Generated Verilog module text.
        """
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
        module_name : str or None
            Optional override for the default module name.
        seed : int
            Seed for Sobol (ignored for Halton).

        Returns
        -------
        str
            Generated Verilog module text.
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
        """Return an instantiation template for ``sc_decorrelator``.

        Parameters
        ----------
        num_streams : int
            Number of parallel bitstream channels.
        stream_width : int
            Bit width of each channel shift register.
        shift_seed : int
            32-bit decorrelation shift seed.

        Returns
        -------
        str
            Verilog instantiation snippet.
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
        """Return an instantiation template for ``sc_edt_controller``.

        Parameters
        ----------
        data_width : int
            Accumulator width in bits.
        margin : int
            16-bit early-decision confidence margin.
        stable_cycles : int
            Required consecutive stable cycles before freeze.

        Returns
        -------
        str
            Verilog instantiation snippet.
        """
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
        """Generate a triple-modular-redundancy wrapper for ``module_name``.

        Parameters
        ----------
        module_name : str
            Inner Verilog module name to replicate.
        inputs : list[tuple[str, int]]
            Input ``(port_name, width)`` specifications.
        outputs : list[tuple[str, int]]
            Output ``(port_name, width)`` specifications.

        Returns
        -------
        str
            Generated TMR Verilog wrapper source.
        """
        from .tmr_wrapper import generate_tmr_wrapper

        return generate_tmr_wrapper(module_name=module_name, inputs=inputs, outputs=outputs)

    def save_to_file(self, path: str) -> None:
        """Write synchronous generated Verilog to ``path``.

        Parameters
        ----------
        path : str
            Destination filesystem path.
        """
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.generate())
        except OSError as exc:
            logger.error("Failed to write Verilog to %s: %s", path, exc)
            raise


def emit_sources_from_ir(ir: Any) -> str:
    """Emit LFSR-16, Sobol-16, and Halton-16 source modules from an IR payload.

    Parameters
    ----------
    ir : Any
        Mapping ``{"nodes": ...}``, object exposing ``nodes``, or sequence of
        node mappings/objects. Non-source nodes are ignored.

    Returns
    -------
    str
        Double-newline-joined Verilog source modules.
    """
    emitted: list[str] = []
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
            emitted.append(Halton16Emitter(module_name=module_name).generate())
    return "\n\n".join(emitted)


def _iter_ir_nodes(ir: Any) -> list[tuple[str | None, Any]]:
    """Iterate IR nodes, yielding ``(node_id, node)`` pairs."""
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
