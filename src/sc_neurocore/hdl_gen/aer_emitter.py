# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Research AER wrapper for HDL generation

from __future__ import annotations

from collections.abc import Mapping
from math import ceil, log2
from numbers import Integral
from typing import Any

from ._ident import sanitize_ident


class AEREmitter:
    """Emit a research-stage AER wrapper around the existing sync HDL path.

    This is intentionally conservative: the compute pipeline remains clocked
    and the output is wrapped in a 4-phase AER-style request/acknowledge
    interface. It is not a QDI async network replacement.
    """

    def __init__(self, module_name: str = "sc_network_async_aer", bus_width: int = 8) -> None:
        self.module_name = sanitize_ident(module_name, context="module name")
        self.bus_width = self._require_positive_int(bus_width, "bus_width")
        self.layers: list[dict[str, Any]] = []

    def add_layer(self, layer_type: str, name: str, params: dict[str, Any]) -> None:
        self.layers.append(
            {
                "type": layer_type,
                "name": sanitize_ident(name, context="layer name"),
                "params": params,
            }
        )

    def generate(self) -> str:
        self._validate_layers()
        layer_widths = self._dense_layer_widths()
        input_width = layer_widths[0][0] if layer_widths else self.bus_width
        spike_width = layer_widths[-1][1] if layer_widths else self.bus_width
        addr_width = max(1, ceil(log2(spike_width)))

        code = f"module {self.module_name} (\n"
        code += "    input wire clk,\n"
        code += "    input wire rst_n,\n"
        code += f"    input wire [{input_width - 1}:0] input_bus,\n"
        code += "    input wire aer_ack,\n"
        code += "    output reg aer_req,\n"
        code += f"    output reg [{addr_width - 1}:0] aer_addr,\n"
        code += f"    output wire [{spike_width - 1}:0] output_bus\n"
        code += ");\n\n"

        code += "    // Research boundary: sync compute path with AER output wrapper.\n"
        code += "    // This is not a full asynchronous micropipeline implementation.\n"
        code += f"    wire [{spike_width - 1}:0] spike_vector;\n"

        for i in range(len(layer_widths) - 1):
            code += f"    wire [{layer_widths[i][1] - 1}:0] layer_{i}_to_{i + 1};\n"
        code += "\n"

        # _validate_layers() above rejects any non-Dense layer, so every layer is
        # Dense and its position ``i`` doubles as the dense-layer index.
        for i, layer in enumerate(self.layers):
            output_bus = "spike_vector" if i == len(layer_widths) - 1 else f"layer_{i}_to_{i + 1}"
            input_bus = "input_bus" if i == 0 else f"layer_{i - 1}_to_{i}"
            code += f"    // Sync layer {i}: {layer['name']}\n"
            code += "    sc_dense_layer_core #(\n"
            code += f"        .NUM_NEURONS({layer['params']['n_neurons']})\n"
            code += f"    ) {layer['name']}_inst (\n"
            code += "        .clk(clk),\n"
            code += "        .rst_n(rst_n),\n"
            code += f"        .input_bus({input_bus}),\n"
            code += f"        .output_bus({output_bus})\n"
            code += "    );\n\n"

        if not self.layers:
            code += f"    assign spike_vector = {spike_width}'b0;\n\n"

        code += "    assign output_bus = spike_vector;\n"
        code += "    wire spike_valid = |spike_vector;\n\n"

        code += f"    function [{addr_width - 1}:0] first_hot_index;\n"
        code += f"        input [{spike_width - 1}:0] vector;\n"
        code += "        integer k;\n"
        code += "        reg found;\n"
        code += "        begin\n"
        code += f"            first_hot_index = {addr_width}'d0;\n"
        code += "            found = 1'b0;\n"
        code += f"            for (k = 0; k < {spike_width}; k = k + 1) begin\n"
        code += "                if (!found && vector[k]) begin\n"
        code += f"                    first_hot_index = k[{addr_width - 1}:0];\n"
        code += "                    found = 1'b1;\n"
        code += "                end\n"
        code += "            end\n"
        code += "        end\n"
        code += "    endfunction\n\n"

        code += f"    wire [{addr_width - 1}:0] encoded_addr = first_hot_index(spike_vector);\n"
        code += f"    reg [{spike_width - 1}:0] event_vector;\n"
        code += f"    reg [{spike_width - 1}:0] acknowledged_vector;\n"
        code += (
            "    wire new_spike_vector = spike_valid && (spike_vector != acknowledged_vector);\n\n"
        )
        code += "    always @(posedge clk or negedge rst_n) begin\n"
        code += "        if (!rst_n) begin\n"
        code += "            aer_req <= 1'b0;\n"
        code += f"            aer_addr <= {addr_width}'d0;\n"
        code += f"            event_vector <= {spike_width}'d0;\n"
        code += f"            acknowledged_vector <= {spike_width}'d0;\n"
        code += "        end else begin\n"
        code += "            if (!spike_valid) begin\n"
        code += f"                acknowledged_vector <= {spike_width}'d0;\n"
        code += "            end\n"
        code += "            if (!aer_req && new_spike_vector) begin\n"
        code += "                aer_req <= 1'b1;\n"
        code += "                aer_addr <= encoded_addr;\n"
        code += "                event_vector <= spike_vector;\n"
        code += "            end else if (aer_req && aer_ack) begin\n"
        code += "                aer_req <= 1'b0;\n"
        code += "                acknowledged_vector <= event_vector;\n"
        code += "            end\n"
        code += "        end\n"
        code += "    end\n\n"
        code += "endmodule\n"
        return code

    @staticmethod
    def _require_positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    def _validate_layers(self) -> None:
        for layer in self.layers:
            if layer["type"] != "Dense":
                raise ValueError(
                    f"unsupported async AER layer type '{layer['type']}' for layer '{layer['name']}'"
                )
            params = layer["params"]
            if "n_neurons" not in params:
                raise ValueError(f"Dense layer '{layer['name']}' requires n_neurons")
            self._require_positive_int(
                params["n_neurons"], f"Dense layer '{layer['name']}' n_neurons"
            )
            for width_name in ("input_width", "output_width"):
                if width_name in params:
                    self._require_positive_int(
                        params[width_name],
                        f"Dense layer '{layer['name']}' {width_name}",
                    )

    def _dense_input_width(self, params: Mapping[str, Any], previous_width: int | None) -> int:
        if "input_width" in params:
            return self._require_positive_int(params["input_width"], "input_width")
        return previous_width if previous_width is not None else self.bus_width

    def _dense_output_width(self, params: Mapping[str, Any]) -> int:
        if "output_width" in params:
            return self._require_positive_int(params["output_width"], "output_width")
        return self._require_positive_int(params["n_neurons"], "n_neurons")

    def _dense_layer_widths(self) -> list[tuple[int, int]]:
        widths: list[tuple[int, int]] = []
        previous_width: int | None = None
        previous_name: str | None = None
        for layer in self.layers:
            params = layer["params"]
            input_width = self._dense_input_width(params, previous_width)
            output_width = self._dense_output_width(params)
            if previous_width is not None and input_width != previous_width:
                raise ValueError(
                    f"{previous_name} -> {layer['name']} width mismatch: "
                    f"{previous_width} output bits cannot drive {input_width} input bits"
                )
            widths.append((input_width, output_width))
            previous_width = output_width
            previous_name = layer["name"]
        return widths
