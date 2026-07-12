# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR/ONNX → FPGA network compiler
"""Combined fixed-point weight-ROM emission for quantised neuron graphs."""

from .quantise_params import QuantisedGraph


def build_weight_rom(
    qgraph: QuantisedGraph,
    *,
    data_width: int = 16,
) -> str:
    """Generate a combined weight ROM for all connections.

    All connection weight matrices are flattened into a single ROM
    addressed by a global index.  Each connection gets a base address
    offset.

    Parameters
    ----------
    qgraph : QuantisedGraph
        Quantised graph with integer weight matrices.
    data_width : int
        Weight data width.

    Returns
    -------
    str
        Verilog weight ROM module source.
    """
    if not qgraph.connections:
        # Empty ROM
        return (
            "// Auto-generated weight ROM (empty — no connections)\n"
            "module sc_nir_weight_rom (\n"
            "    input  wire [0:0] addr,\n"
            f"    output wire signed [{data_width - 1}:0] data\n"
            ");\n"
            f"    assign data = {data_width}'sd0;\n"
            "endmodule\n"
        )

    # Flatten all weights into a single list
    all_weights: list[int] = []
    conn_offsets: list[tuple[str, str, int, int]] = []  # src, dst, offset, count

    for conn in qgraph.connections:
        offset = len(all_weights)
        flat = conn.weights.flatten().tolist()
        count = len(flat)
        all_weights.extend(int(w) for w in flat)
        conn_offsets.append((conn.src, conn.dst, offset, count))

    total = len(all_weights)
    addr_w = max(1, (total - 1).bit_length())

    lines = [
        "// Auto-generated combined weight ROM",
        "// SC-NeuroCore NIR → FPGA compiler",
        f"// Total entries: {total}, Address width: {addr_w}",
        "",
    ]

    # Connection offset comments
    for src, dst, offset, count in conn_offsets:
        lines.append(f"// {src} → {dst}: offset={offset}, count={count}")
    lines.append("")

    lines.extend(
        [
            "module sc_nir_weight_rom (",
            f"    input  wire [{addr_w - 1}:0] addr,",
            f"    output reg  signed [{data_width - 1}:0] data",
            ");",
            "",
            "    always @(*) begin",
            "        case (addr)",
        ]
    )

    mask = (1 << data_width) - 1
    for i, w in enumerate(all_weights):
        val = w & mask
        lines.append(f"            {addr_w}'d{i}: data = {data_width}'sh{val:0{data_width // 4}x};")

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
