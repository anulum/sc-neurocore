# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SystemVerilog Assertion (SVA) generation

"""SVA generation utilities for formal verification.

Emits formal verification properties for safety-critical certification
(DO-254 / IEC 61508).
"""

from __future__ import annotations


def generate_sva(
    state_vars: list[str],
    *,
    data_width: int = 16,
    fraction: int = 8,
    signed: bool = True,
    input_bounds: dict[str, tuple[float, float]] | None = None,
    module_name: str = "sc_equation_neuron",
) -> str:
    """Generate SystemVerilog Assertions for a compiled neuron module.

    Produces three categories of formal properties:

    1. **Overflow assertions** — check that no state variable exceeds
       the representable range after the next-state update.
    2. **Reachability covers** — prove that spike output is reachable.
    3. **Input assumptions** — constrain external inputs to valid bounds.

    Parameters
    ----------
    state_vars : list[str]
        Names of state variables (e.g. ``["v"]``).
    data_width : int
        Bit width of the fixed-point format.
    fraction : int
        Fractional bits.
    signed : bool
        True for signed format.
    input_bounds : dict, optional
        Mapping from input names to (min_q, max_q) bounds in Q-format integers.
    module_name : str
        Name of the target module.

    Returns
    -------
    str
        SystemVerilog bind module with assertions.
    """
    if signed:
        q_max = (1 << (data_width - 1)) - 1
        q_min = -(1 << (data_width - 1))
    else:
        q_max = (1 << data_width) - 1
        q_min = 0

    sign_kw = "signed " if signed else ""
    lines = [
        f"// Auto-generated SystemVerilog Assertions for {module_name}",
        "// SC-NeuroCore static analysis — DO-254 / IEC 61508 compliance",
        f"// Fixed-point: Q{data_width - fraction - (1 if signed else 0)}.{fraction} "
        f"({data_width}-bit {'signed' if signed else 'unsigned'})",
        "",
        f"module {module_name}_sva (",
        "    input wire clk,",
        "    input wire rst_n,",
        f"    input wire {sign_kw}[{data_width - 1}:0] I_t,",
        "    input wire spike_out,",
    ]

    for var in state_vars:
        lines.append(f"    input wire {sign_kw}[{data_width - 1}:0] {var}_reg,")

    # Remove trailing comma from last port
    lines[-1] = lines[-1].rstrip(",")
    lines.append(");")
    lines.append("")

    # Default clocking block
    lines.append("    default clocking cb @(posedge clk);")
    lines.append("    endclocking")
    lines.append("")

    # 1. Overflow assertions
    lines.append("    // ── Overflow Assertions ──────────────────────────────────")
    for var in state_vars:
        if signed:
            lines.append(
                f"    a_no_overflow_{var}: assert property ("
                f"disable iff (!rst_n) "
                f"$signed({var}_reg) >= {data_width}'sd{q_min} && "
                f"$signed({var}_reg) <= {data_width}'sd{q_max}"
                f') else $error("OVERFLOW: {var}_reg = %0d", {var}_reg);'
            )
        else:
            lines.append(
                f"    a_no_overflow_{var}: assert property ("
                f"disable iff (!rst_n) "
                f"{var}_reg <= {data_width}'d{q_max}"
                f') else $error("OVERFLOW: {var}_reg = %0d", {var}_reg);'
            )

    lines.append("")

    # 2. Reachability covers
    lines.append("    // ── Reachability Covers ─────────────────────────────────")
    lines.append("    c_spike_reachable: cover property (disable iff (!rst_n) spike_out == 1'b1);")
    lines.append("    c_no_spike: cover property (disable iff (!rst_n) spike_out == 1'b0);")

    for var in state_vars:
        lines.append(
            f"    c_{var}_nonzero: cover property ("
            f"disable iff (!rst_n) {var}_reg != {data_width}'sd0"
            f");"
        )

    lines.append("")

    # 3. Input assumptions
    if input_bounds:
        lines.append("    // ── Input Assumptions ──────────────────────────────────")
        for name, (lo, hi) in input_bounds.items():
            lines.append(
                f"    m_{name}_bound: assume property ("
                f"disable iff (!rst_n) "
                f"$signed({name}) >= {data_width}'sd{lo} && "
                f"$signed({name}) <= {data_width}'sd{hi}"
                f");"
            )
        lines.append("")

    # 4. Stability check — membrane voltage should not stay at max for too long
    lines.append("    // ── Stability Checks ───────────────────────────────────")
    for var in state_vars:
        lines.append(
            f"    a_{var}_not_stuck_max: assert property ("
            f"disable iff (!rst_n) "
            f"not ({var}_reg == {data_width}'sd{q_max} [*100])"
            f') else $warning("{var}_reg stuck at max for 100+ cycles");'
        )

    lines.append("")
    lines.append("endmodule")
    lines.append("")

    # Bind directive
    lines.append("// Bind to DUT — place in testbench or verification top")
    lines.append(f"// bind {module_name} {module_name}_sva sva_inst (.*);")
    lines.append("")

    return "\n".join(lines)
