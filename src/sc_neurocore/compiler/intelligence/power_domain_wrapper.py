# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Power domain wrapper generator

"""Clock and power gating wrapper generation for edge deployment."""

from __future__ import annotations


def generate_power_domain_wrapper(
    module_name: str,
    *,
    data_width: int = 16,
    state_vars: list[str] | None = None,
    always_on_signals: list[str] | None = None,
    wakeup_cycles: int = 4,
) -> str:
    """Generate a clock/power gating wrapper for always-on edge deployment.

    Creates a wrapper module with ICG and power-down state retention.

    Parameters
    ----------
    module_name : str
        Inner neuron module name.
    data_width : int
        Data width.
    state_vars : list[str], optional
        State variables to retain. Defaults to ``["v"]``.
    always_on_signals : list[str], optional
        Signals kept in the always-on domain. Defaults to ``["spike_out"]``.
    wakeup_cycles : int
        Clock cycles required to exit power-down.

    Returns
    -------
    str
        Synthesisable Verilog power domain wrapper.
    """
    if state_vars is None:
        state_vars = ["v"]
    if always_on_signals is None:
        always_on_signals = ["spike_out"]

    w = data_width
    pg_name = f"{module_name}_pg"
    wk_bits = max(1, (wakeup_cycles - 1).bit_length())

    lines = [
        f"// Auto-generated power domain wrapper for {module_name}",
        "// SC-NeuroCore — clock/power gating for ultra-low-power edge",
        f"// Wakeup latency: {wakeup_cycles} cycles",
        "",
        f"module {pg_name} (",
        "    input  wire clk,",
        "    input  wire rst,",
        "    input  wire en,",
        "    input  wire power_down,     // Active-high power-down request",
        f"    input  wire signed [{w - 1}:0] I_t,",
    ]
    for sv in state_vars:
        lines.append(f"    output reg  signed [{w - 1}:0] {sv}_out,")
    for sig in always_on_signals:
        lines.append(f"    output wire {sig},")
    lines.append("    output wire power_state       // 0=active, 1=power-down")
    lines.append(");")
    lines.append("")

    # ICG cell
    lines.extend(
        [
            "    // ── Integrated Clock Gating ──",
            "    wire gated_clk;",
            "    reg  clk_enable;",
            "    always @(negedge clk)",
            "        clk_enable <= en & ~power_down;",
            "    assign gated_clk = clk & clk_enable;",
            "",
        ]
    )

    # Wakeup counter
    lines.extend(
        [
            "    // ── Wakeup sequencer ──",
            f"    reg [{wk_bits - 1}:0] wakeup_cnt;",
            "    reg  active;",
            "    always @(posedge clk or posedge rst) begin",
            "        if (rst) begin",
            "            wakeup_cnt <= 0;",
            "            active <= 0;",
            "        end else if (power_down) begin",
            "            wakeup_cnt <= 0;",
            "            active <= 0;",
            "        end else if (!active) begin",
            f"            if (wakeup_cnt == {wakeup_cycles - 1})",
            "                active <= 1;",
            "            else",
            "                wakeup_cnt <= wakeup_cnt + 1;",
            "        end",
            "    end",
            "    assign power_state = ~active;",
            "",
        ]
    )

    # Inner module instance
    lines.extend(
        [
            "    // ── Inner neuron (gated clock domain) ──",
        ]
    )
    for sv in state_vars:
        lines.append(f"    wire signed [{w - 1}:0] {sv}_inner;")
    lines.extend(
        [
            "    wire spike_inner;",
            "",
            f"    {module_name} core (",
            "        .clk(gated_clk), .rst(rst), .en(active),",
            "        .I_t(I_t),",
        ]
    )
    for sv in state_vars:
        lines.append(f"        .{sv}_next({sv}_inner),")
    lines.extend(
        [
            "        .spike_out(spike_inner)",
            "    );",
            "",
        ]
    )

    # State retention
    lines.extend(
        [
            "    // ── State retention latches ──",
            "    always @(posedge clk or posedge rst) begin",
            "        if (rst) begin",
        ]
    )
    for sv in state_vars:
        lines.append(f"            {sv}_out <= 0;")
    lines.extend(
        [
            "        end else if (active) begin",
        ]
    )
    for sv in state_vars:
        lines.append(f"            {sv}_out <= {sv}_inner;")
    lines.extend(
        [
            "        end",
            "        // Else: retain previous value (power-down)",
            "    end",
            "",
        ]
    )

    # Always-on spike detection
    lines.extend(
        [
            "    // ── Always-on domain (ungated) ──",
            "    assign spike_out = active ? spike_inner : 1'b0;",
            "",
            "endmodule",
        ]
    )

    return "\n".join(lines)
