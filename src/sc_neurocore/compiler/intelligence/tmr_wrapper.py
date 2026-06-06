# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — TMR wrapper generator

"""Triple Modular Redundancy (TMR) wrapper generation for safety-critical RTL."""

from __future__ import annotations


def generate_tmr_wrapper(
    module_name: str,
    *,
    data_width: int = 16,
    state_vars: list[str] | None = None,
    voter: str = "majority",
) -> str:
    """Generate a Triple Modular Redundancy wrapper for any neuron module.

    Instantiates three copies of the target module and a majority voter
    to mask Single Event Upsets (SEUs).

    Parameters
    ----------
    module_name : str
        Name of the inner neuron module to wrap.
    data_width : int
        Data width of the inner module.
    state_vars : list[str], optional
        State variable names for output voting. Defaults to ``["v"]``.
    voter : str
        Voter type: ``"majority"`` (2-of-3) or ``"median"`` (middle value).

    Returns
    -------
    str
        Synthesisable Verilog TMR wrapper module.
    """
    if state_vars is None:
        state_vars = ["v"]

    w = data_width
    tmr_name = f"{module_name}_tmr"

    lines = [
        f"// Auto-generated TMR wrapper for {module_name}",
        "// SC-NeuroCore SEU mitigation — Triple Modular Redundancy",
        f"// Voter: {voter} | DO-254 DAL-A / IEC 61508 SIL-4",
        "// IMPORTANT: Place each instance in a separate region (PBLOCK)",
        "",
        f"module {tmr_name} (",
        "    input  wire clk,",
        "    input  wire rst,",
        "    input  wire en,",
        f"    input  wire signed [{w - 1}:0] I_t,",
    ]
    for sv in state_vars:
        lines.append(f"    output wire signed [{w - 1}:0] {sv}_voted,")
    lines.append("    output wire spike_out,")
    lines.append("    output wire seu_detected")
    lines.append(");")
    lines.append("")

    # Instantiate three copies
    for i in range(3):
        lines.append(f"    // ── Instance {chr(65 + i)} ──")
        for sv in state_vars:
            lines.append(f"    wire signed [{w - 1}:0] {sv}_{chr(97 + i)};")
        lines.append(f"    wire spike_{chr(97 + i)};")
        lines.append("")
        lines.append(f"    {module_name} inst_{chr(97 + i)} (")
        lines.append("        .clk(clk), .rst(rst), .en(en), .I_t(I_t),")
        for sv in state_vars:
            lines.append(f"        .{sv}_next({sv}_{chr(97 + i)}),")
        lines.append(f"        .spike_out(spike_{chr(97 + i)})")
        lines.append("    );")
        lines.append("")

    # Majority voter
    lines.append("    // ── Majority Voter ──")
    if voter == "majority":
        for sv in state_vars:
            a, b, c = f"{sv}_a", f"{sv}_b", f"{sv}_c"
            lines.extend(
                [
                    f"    assign {sv}_voted = ({a} & {b}) | ({b} & {c}) | ({a} & {c});",
                ]
            )
        lines.append(
            "    assign spike_out = (spike_a & spike_b) | "
            "(spike_b & spike_c) | (spike_a & spike_c);"
        )
    else:  # median
        for sv in state_vars:
            a, b, c = f"{sv}_a", f"{sv}_b", f"{sv}_c"
            lines.extend(
                [
                    "    // Median: sort three values, pick middle",
                    f"    wire signed [{w - 1}:0] {sv}_min = "
                    f"($signed({a}) < $signed({b})) ? "
                    f"(($signed({a}) < $signed({c})) ? {a} : {c}) : "
                    f"(($signed({b}) < $signed({c})) ? {b} : {c});",
                    f"    wire signed [{w - 1}:0] {sv}_max = "
                    f"($signed({a}) > $signed({b})) ? "
                    f"(($signed({a}) > $signed({c})) ? {a} : {c}) : "
                    f"(($signed({b}) > $signed({c})) ? {b} : {c});",
                    f"    assign {sv}_voted = {a} + {b} + {c} - {sv}_min - {sv}_max;",
                ]
            )
        lines.append(
            "    assign spike_out = (spike_a & spike_b) | "
            "(spike_b & spike_c) | (spike_a & spike_c);"
        )

    # SEU detection: any mismatch
    mismatch_terms = []
    for sv in state_vars:
        a, b, c = f"{sv}_a", f"{sv}_b", f"{sv}_c"
        mismatch_terms.append(f"({a} != {b})")
        mismatch_terms.append(f"({b} != {c})")
    mismatch_terms.append("(spike_a != spike_b)")
    mismatch_terms.append("(spike_b != spike_c)")
    lines.append(f"    assign seu_detected = {' | '.join(mismatch_terms)};")
    lines.append("")
    lines.append("endmodule")

    return "\n".join(lines)
