# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Weight ROM generation

"""Weight ROM generation utilities for synaptic connections.

Generates synthesisable Verilog ROMs or memory initialisation files
(.coe/.mif) for FPGA Block RAMs.
"""

from __future__ import annotations


def generate_weight_rom(
    weights: list[list[int]],
    module_name: str = "sc_weight_rom",
    *,
    data_width: int = 16,
    output_format: str = "verilog",
) -> str:
    """Generate a weight ROM for synaptic connections.

    Produces either a Verilog ROM module or a Xilinx ``.coe`` / Intel
    ``.mif`` memory initialisation file for BRAM-based weight storage.

    Parameters
    ----------
    weights : list[list[int]]
        2D weight matrix [src_neuron][dst_neuron] in Q-format integers.
    module_name : str
        ROM module name.
    data_width : int
        Weight bit width.
    output_format : str
        ``"verilog"`` (synthesisable ROM), ``"coe"`` (Xilinx), ``"mif"`` (Intel).

    Returns
    -------
    str
        Weight ROM in the specified format.
    """
    n_src = len(weights)
    n_dst = len(weights[0]) if weights else 0
    total_entries = n_src * n_dst
    addr_w = max(1, (total_entries - 1).bit_length())

    flat_weights = [w for row in weights for w in row]

    if output_format == "coe":
        lines = [
            "; Auto-generated Xilinx .coe weight file",
            f"; SC-NeuroCore: {n_src}×{n_dst} synaptic weights",
            "memory_initialization_radix=16;",
            "memory_initialization_vector=",
        ]
        for i, w in enumerate(flat_weights):
            val = w & ((1 << data_width) - 1)
            sep = ";" if i == len(flat_weights) - 1 else ","
            lines.append(f"{val:0{data_width // 4}x}{sep}")
        return "\n".join(lines)

    elif output_format == "mif":
        lines = [
            "-- Auto-generated Intel .mif weight file",
            f"-- SC-NeuroCore: {n_src}×{n_dst} synaptic weights",
            f"WIDTH={data_width};",
            f"DEPTH={total_entries};",
            "ADDRESS_RADIX=UNS;",
            "DATA_RADIX=HEX;",
            "CONTENT BEGIN",
        ]
        for i, w in enumerate(flat_weights):
            val = w & ((1 << data_width) - 1)
            lines.append(f"  {i} : {val:0{data_width // 4}x};")
        lines.append("END;")
        return "\n".join(lines)

    elif output_format == "verilog":
        lines = [
            f"// Auto-generated weight ROM: {module_name}",
            f"// SC-NeuroCore: {n_src}×{n_dst} synaptic weights",
            "",
            f"module {module_name} (",
            f"    input  wire [{addr_w - 1}:0] addr,",
            f"    output reg  signed [{data_width - 1}:0] data",
            ");",
            "",
            "    always @(*) begin",
            "        case (addr)",
        ]
        for i, w in enumerate(flat_weights):
            val = w & ((1 << data_width) - 1)
            lines.append(
                f"            {addr_w}'d{i}: data = {data_width}'sh{val:0{data_width // 4}x};"
            )
        lines.extend(
            [
                f"            default: data = {data_width}'sd0;",
                "        endcase",
                "    end",
                "",
                "endmodule",
            ]
        )
        return "\n".join(lines)
    raise ValueError(f"Unsupported weight ROM format: {output_format!r}")
