# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pipeline register wrapper

"""Pipelined wrapper generation for high-frequency FPGA targets."""

from __future__ import annotations


def generate_pipeline_wrapper(
    module_name: str,
    equations: dict[str, str],
    *,
    data_width: int = 16,
    target: str = "artix7",
    stages: int | None = None,
) -> str:
    """Generate a pipelined wrapper that inserts register stages.

    Auto-computes the critical path depth and required pipeline stages.

    Parameters
    ----------
    module_name : str
        Inner neuron module name.
    equations : dict[str, str]
        ODE equations.
    data_width : int
        Data width.
    target : str
        Target platform name.
    stages : int, optional
        Override pipeline stages.

    Returns
    -------
    str
        Synthesisable Verilog pipeline wrapper.
    """
    from ..static_analysis import critical_path_depth, pipeline_stages_needed
    from ...platforms import get_profile

    profile = get_profile(target)
    freq = profile.max_freq_mhz or 100

    max_depth = 0
    for _sv, expr in equations.items():
        d = critical_path_depth(expr)
        max_depth = max(max_depth, d)

    if stages is None:
        stages = pipeline_stages_needed(max_depth, freq)

    if stages == 0:
        stages = 1

    w = data_width
    pipe_name = f"{module_name}_pipe"

    lines = [
        f"// Auto-generated pipeline wrapper for {module_name}",
        f"// SC-NeuroCore — {stages}-stage pipeline for {freq} MHz",
        f"// Critical path depth: {max_depth} DSP blocks",
        "",
        f"module {pipe_name} (",
        "    input  wire clk,",
        "    input  wire rst,",
        "    input  wire en,",
        "    input  wire valid_in,",
        f"    input  wire signed [{w - 1}:0] I_t,",
        f"    output wire signed [{w - 1}:0] v_out,",
        "    output wire spike_out,",
        "    output wire valid_out,",
        f"    output wire [{stages.bit_length() - 1}:0] latency",
        ");",
        "",
        "    // Pipeline latency (constant)",
        f"    assign latency = {stages};",
        "",
    ]

    # Input pipeline registers
    lines.append("    // ── Input pipeline registers ──")
    for s in range(stages):
        lines.append(f"    reg signed [{w - 1}:0] I_pipe_{s};")
    lines.append("")

    lines.extend(
        [
            "    always @(posedge clk or posedge rst) begin",
            "        if (rst) begin",
        ]
    )
    for s in range(stages):
        lines.append(f"            I_pipe_{s} <= 0;")
    lines.extend(
        [
            "        end else if (en) begin",
            "            I_pipe_0 <= I_t;",
        ]
    )
    for s in range(1, stages):
        lines.append(f"            I_pipe_{s} <= I_pipe_{s - 1};")
    lines.extend(
        [
            "        end",
            "    end",
            "",
        ]
    )

    # Valid pipeline
    lines.extend(
        [
            "    // ── Valid pipeline ──",
            f"    reg [{stages - 1}:0] valid_pipe;",
            "    always @(posedge clk or posedge rst) begin",
            "        if (rst)",
            "            valid_pipe <= 0;",
            "        else if (en)",
        ]
    )
    if stages == 1:
        lines.append("            valid_pipe[0] <= valid_in;")
    else:
        lines.append(f"            valid_pipe <= {{valid_pipe[{stages - 2}:0], valid_in}};")
    lines.extend(
        [
            "    end",
            f"    assign valid_out = valid_pipe[{stages - 1}];",
            "",
        ]
    )

    # Inner module instantiation
    lines.extend(
        [
            "    // ── Inner neuron (combinational) ──",
            f"    wire signed [{w - 1}:0] v_comb;",
            "    wire spike_comb;",
            "",
            f"    {module_name} core (",
            "        .clk(clk), .rst(rst), .en(en),",
            f"        .I_t(I_pipe_{stages - 1}),",
            "        .v_next(v_comb),",
            "        .spike_out(spike_comb)",
            "    );",
            "",
        ]
    )

    # Output register
    lines.extend(
        [
            "    // ── Output register ──",
            f"    reg signed [{w - 1}:0] v_reg;",
            "    reg spike_reg;",
            "    always @(posedge clk or posedge rst) begin",
            "        if (rst) begin",
            "            v_reg <= 0;",
            "            spike_reg <= 0;",
            "        end else if (en) begin",
            "            v_reg <= v_comb;",
            "            spike_reg <= spike_comb;",
            "        end",
            "    end",
            "    assign v_out = v_reg;",
            "    assign spike_out = spike_reg;",
            "",
            "endmodule",
        ]
    )

    return "\n".join(lines)
