# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — HLS C++ export

"""High-Level Synthesis (HLS) C++ export.

Lowers ODE neuron equations to synthesisable ``ap_fixed`` C++ for Xilinx Vitis
HLS or Siemens Catapult. Each equation value is treated as a derivative
``d<var>/dt`` (matching the Verilog backend) and integrated with an explicit
Euler step ``<var>_next = <var> + dt * d<var>``; the fixed-point arithmetic is
carried by the ``ap_fixed`` type. Expressions are parsed and lowered through
:class:`sc_neurocore.compiler.c_expr_emitter.CExprEmitter`, so Python operators
and transcendental calls become valid C++ (rather than being embedded verbatim),
and any free identifiers become function inputs so the generated unit compiles.
"""

from __future__ import annotations

from ..c_expr_emitter import emit_c_expr


def _preamble(module_name: str, data_width: int, fraction: int, hls_tool: str) -> list[str]:
    """Emit the include guard, headers, and fixed-point typedef."""
    int_bits = data_width - fraction
    guard = module_name.upper()
    return [
        f"// Auto-generated HLS C++ for {module_name}",
        f"// SC-NeuroCore — {hls_tool.upper()} HLS export",
        f"// Q{int_bits}.{fraction} fixed-point ({data_width}-bit)",
        "",
        f"#ifndef {guard}_HLS_H",
        f"#define {guard}_HLS_H",
        "",
        '#include "ap_fixed.h"',
        '#include "hls_math.h"',
        "",
        f"typedef ap_fixed<{data_width},{int_bits}> fp_t;",
        "",
    ]


def _helpers(used: set[str]) -> list[str]:
    """Emit inline fixed-point helpers for any non-library transcendentals used."""
    lines: list[str] = []
    if "sc_sigmoid" in used:
        lines += [
            "static inline fp_t sc_sigmoid(fp_t x) {",
            "    return fp_t(1) / (fp_t(1) + hls::exp(-x));",
            "}",
            "",
        ]
    if "sc_exprel" in used:
        lines += [
            "// exprel(x) = (exp(x) - 1) / x, with the removable-singularity limit 1 at x = 0.",
            "static inline fp_t sc_exprel(fp_t x) {",
            "    return (hls::abs(x) < fp_t(0.001)) ? fp_t(1) : (hls::exp(x) - fp_t(1)) / x;",
            "}",
            "",
        ]
    return lines


def generate_hls_cpp(
    module_name: str,
    equations: dict[str, str],
    *,
    data_width: int = 16,
    fraction: int = 8,
    hls_tool: str = "vitis",
    dt: float = 1.0,
    threshold: float = 1.0,
) -> str:
    """Translate compiled neuron equations to Vitis/Catapult HLS C++.

    Generates a synthesisable ``ap_fixed`` C++ function with ``#pragma HLS``
    directives for Xilinx Vitis HLS or Siemens Catapult. Each equation is a
    derivative that is Euler-integrated (``<var>_next = <var> + dt * d<var>``);
    the first state variable is treated as the membrane potential and reset by
    subtracting the threshold when it spikes. Free identifiers in the equations
    become function inputs so the generated unit is self-contained.

    Parameters
    ----------
    module_name : str
        Function/module name.
    equations : dict
        Mapping ``state_var -> derivative expression`` (Python syntax).
    data_width : int
        Fixed-point total width.
    fraction : int
        Fractional bits.
    hls_tool : str
        ``"vitis"`` or ``"catapult"``.
    dt : float
        Euler integration timestep.
    threshold : float
        Membrane spike threshold.

    Returns
    -------
    str
        Complete HLS C++ source file.
    """
    guard = module_name.upper()
    state_vars = set(equations)

    # Lower every derivative expression once, collecting free variables (in
    # first-seen order across equations) and which inline helpers are needed.
    lowered: dict[str, str] = {}
    free_vars: list[str] = []
    used_helpers: set[str] = set()
    for sv, expr in equations.items():
        c_expr, expr_free = emit_c_expr(expr, state_vars)
        lowered[sv] = c_expr
        for fv in expr_free:
            if fv not in free_vars:
                free_vars.append(fv)
        if "sc_sigmoid(" in c_expr:
            used_helpers.add("sc_sigmoid")
        if "sc_exprel(" in c_expr:
            used_helpers.add("sc_exprel")

    lines = _preamble(module_name, data_width, fraction, hls_tool)
    lines += _helpers(used_helpers)

    # State struct.
    lines.append(f"struct {module_name}_state {{")
    for sv in equations:
        lines.append(f"    fp_t {sv};")
    lines += ["    bool spike;", "};", ""]

    # Function signature: input current, timestep, free-variable inputs, then
    # state references (by-reference for in-place update) and the spike output.
    lines.append(f"void {module_name}(")
    params = ["    fp_t I_t,", "    fp_t dt,"]
    params += [f"    fp_t {fv}," for fv in free_vars]
    params += [f"    fp_t &{sv}," for sv in equations]
    params.append("    bool &spike_out")
    lines += params
    lines.append(") {")

    if hls_tool == "vitis":
        lines.append("    #pragma HLS PIPELINE II=1")
        lines.append("    #pragma HLS INTERFACE ap_ctrl_none port=return")
    else:  # catapult
        lines.append("    // Catapult: pipeline directive applied at synthesis")
    lines.append("")

    # Euler integration of each derivative.
    for sv in equations:
        lines.append(f"    fp_t d_{sv} = {lowered[sv]};")
        lines.append(f"    fp_t {sv}_next = {sv} + dt * d_{sv};")
    lines.append("")

    # Threshold detection on the membrane (first) state variable.
    membrane = next(iter(equations))
    lines += [
        "    // Threshold detection on the membrane state variable",
        f"    const fp_t V_THRESH = fp_t({threshold!r});",
        f"    spike_out = ({membrane}_next > V_THRESH);",
        "",
    ]

    # Writeback: the membrane resets by subtracting the threshold on a spike.
    for sv in equations:
        if sv == membrane:
            lines.append(f"    {sv} = spike_out ? ({sv}_next - V_THRESH) : {sv}_next;")
        else:
            lines.append(f"    {sv} = {sv}_next;")

    lines += ["}", "", f"#endif // {guard}_HLS_H"]
    return "\n".join(lines)
