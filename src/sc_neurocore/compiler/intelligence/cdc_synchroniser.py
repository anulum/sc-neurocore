# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CDC synchroniser

"""CDC (Clock Domain Crossing) synchroniser generation."""

from __future__ import annotations


def generate_cdc_synchroniser(
    signal_name: str,
    *,
    width: int = 1,
    stages: int = 2,
    src_clock: str = "clk_src",
    dst_clock: str = "clk_dst",
) -> str:
    """Generate a CDC (Clock Domain Crossing) synchroniser in Verilog.

    Uses a multi-stage register chain to safely transfer signals between
    clock domains.

    Parameters
    ----------
    signal_name : str
        Name of the signal being synchronised.
    width : int
        Bit width (1 for single-bit CDC).
    stages : int
        Number of synchroniser stages (2 minimum, 3 for MTBF).
    src_clock : str
        Source clock name.
    dst_clock : str
        Destination clock name.

    Returns
    -------
    str
        Verilog CDC synchroniser module.
    """
    module_name = f"cdc_sync_{signal_name}"
    w = f"[{width - 1}:0] " if width > 1 else ""

    lines = [
        f"// Auto-generated CDC synchroniser for '{signal_name}'",
        "// SC-NeuroCore multi-clock domain support",
        f"// Stages: {stages}, Width: {width}-bit",
        "",
        '(* ASYNC_REG = \"TRUE\" *)  // Xilinx: place in same slice',
        f"module {module_name} (",
        f"    input  wire         {src_clock},",
        f"    input  wire         {dst_clock},",
        "    input  wire         rst,",
        f"    input  wire {w}{signal_name}_in,",
        f"    output wire {w}{signal_name}_out",
        ");",
        "",
    ]

    # Synchroniser chain
    for i in range(stages):
        lines.append(f'    (* ASYNC_REG = \"TRUE\" *) reg {w}sync_r{i};')

    lines.extend(
        [
            "",
            f"    always @(posedge {dst_clock} or posedge rst) begin",
            "        if (rst) begin",
        ]
    )

    for i in range(stages):
        lines.append(f"            sync_r{i} <= {width}'d0;")

    lines.extend(
        [
            "        end else begin",
            f"            sync_r0 <= {signal_name}_in;",
        ]
    )

    for i in range(1, stages):
        lines.append(f"            sync_r{i} <= sync_r{i - 1};")

    lines.extend(
        [
            "        end",
            "    end",
            "",
            f"    assign {signal_name}_out = sync_r{stages - 1};",
            "",
            "endmodule",
            "",
        ]
    )

    return "\n".join(lines)
