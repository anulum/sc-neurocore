# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Research AER wrapper for HDL generation

from __future__ import annotations

from typing import Any

from ._ident import sanitize_ident


class AEREmitter:
    """Emit a research-stage AER wrapper around the existing sync HDL path.

    This is intentionally conservative: the compute pipeline remains clocked
    and the output is wrapped in a 4-phase AER-style request/acknowledge
    interface. It is not a QDI async network replacement.
    """

    def __init__(self, module_name: str = "sc_network_async_aer") -> None:
        self.module_name = sanitize_ident(module_name, context="module name")
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
        code = f"module {self.module_name} (\n"
        code += "    input wire clk,\n"
        code += "    input wire rst_n,\n"
        code += "    input wire [7:0] input_bus,\n"
        code += "    input wire aer_ack,\n"
        code += "    output reg aer_req,\n"
        code += "    output reg [7:0] aer_addr,\n"
        code += "    output wire [7:0] output_bus\n"
        code += ");\n\n"

        code += "    // Research boundary: sync compute path with AER output wrapper.\n"
        code += "    // This is not a full asynchronous micropipeline implementation.\n"
        code += "    wire [7:0] spike_vector;\n"

        for i in range(len(self.layers) - 1):
            code += f"    wire [7:0] layer_{i}_to_{i + 1};\n"
        code += "\n"

        for i, layer in enumerate(self.layers):
            if layer["type"] != "Dense":
                continue

            output_bus = "spike_vector" if i == len(self.layers) - 1 else f"layer_{i}_to_{i + 1}"
            input_bus = "input_bus" if i == 0 else f"layer_{i - 1}_to_{i}"
            code += f"    // Sync layer {i}: {layer['name']}\n"
            code += "    sc_dense_layer_core #(\n"
            code += f"        .NUM_NEURONS({layer['params'].get('n_neurons', 10)})\n"
            code += f"    ) {layer['name']}_inst (\n"
            code += "        .clk(clk),\n"
            code += "        .rst_n(rst_n),\n"
            code += f"        .input_bus({input_bus}),\n"
            code += f"        .output_bus({output_bus})\n"
            code += "    );\n\n"

        if not self.layers:
            code += "    assign spike_vector = 8'b0;\n\n"

        code += "    assign output_bus = spike_vector;\n"
        code += "    wire spike_valid = |spike_vector;\n\n"

        code += "    function [7:0] first_hot_index;\n"
        code += "        input [7:0] vector;\n"
        code += "        begin\n"
        code += "            casex (vector)\n"
        code += "                8'b???????1: first_hot_index = 8'd0;\n"
        code += "                8'b??????10: first_hot_index = 8'd1;\n"
        code += "                8'b?????100: first_hot_index = 8'd2;\n"
        code += "                8'b????1000: first_hot_index = 8'd3;\n"
        code += "                8'b???10000: first_hot_index = 8'd4;\n"
        code += "                8'b??100000: first_hot_index = 8'd5;\n"
        code += "                8'b?1000000: first_hot_index = 8'd6;\n"
        code += "                8'b10000000: first_hot_index = 8'd7;\n"
        code += "                default: first_hot_index = 8'd0;\n"
        code += "            endcase\n"
        code += "        end\n"
        code += "    endfunction\n\n"

        code += "    wire [7:0] encoded_addr = first_hot_index(spike_vector);\n\n"
        code += "    always @(posedge clk or negedge rst_n) begin\n"
        code += "        if (!rst_n) begin\n"
        code += "            aer_req <= 1'b0;\n"
        code += "            aer_addr <= 8'd0;\n"
        code += "        end else begin\n"
        code += "            if (!aer_req && spike_valid) begin\n"
        code += "                aer_req <= 1'b1;\n"
        code += "                aer_addr <= encoded_addr;\n"
        code += "            end else if (aer_req && aer_ack) begin\n"
        code += "                aer_req <= 1'b0;\n"
        code += "            end\n"
        code += "        end\n"
        code += "    end\n\n"
        code += "endmodule\n"
        return code
