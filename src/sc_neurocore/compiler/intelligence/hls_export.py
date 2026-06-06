# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — HLS C++ export

"""High-Level Synthesis (HLS) C++ export utilities.

Translates neuron equations to synthesisable C++ for Xilinx Vitis HLS
or Siemens Catapult HLS.
"""

from __future__ import annotations


def generate_hls_cpp(
    module_name: str,
    equations: dict[str, str],
    *,
    data_width: int = 16,
    fraction: int = 8,
    hls_tool: str = "vitis",
) -> str:
    """Translate compiled neuron equations to Vitis/Catapult HLS C++.

    Generates a synthesisable C++ function with ``#pragma HLS`` directives
    for Xilinx Vitis HLS or Siemens Catapult. Enables HW/SW co-design
    workflows where the neuron runs as an HLS IP block alongside a
    MicroBlaze or RISC-V soft processor.

    Parameters
    ----------
    module_name : str
        Function/module name.
    equations : dict[str, str]
        ODE equations (state_var → C-style expression).
    data_width : int
        Fixed-point total width.
    fraction : int
        Fractional bits.
    hls_tool : str
        ``"vitis"`` or ``"catapult"``.

    Returns
    -------
    str
        Complete HLS C++ source file.
    """
    int_bits = data_width - fraction
    guard = module_name.upper()
    ap_type = f"ap_fixed<{data_width},{int_bits}>"

    lines = [
        f"// Auto-generated HLS C++ for {module_name}",
        f"// SC-NeuroCore — {hls_tool.upper()} HLS export",
        f"// Q{int_bits}.{fraction} fixed-point ({data_width}-bit)",
        "",
        f"#ifndef {guard}_HLS_H",
        f"#define {guard}_HLS_H",
        "",
        '#include "ap_fixed.h"',
        "",
        f"typedef {ap_type} fp_t;",
        "",
    ]

    # Struct for state variables
    lines.extend(
        [
            f"struct {module_name}_state {{",
        ]
    )
    for sv in equations:
        lines.append(f"    fp_t {sv};")
    lines.extend(
        [
            "    bool spike;",
            "};",
            "",
        ]
    )

    # Main function
    lines.extend(
        [
            f"void {module_name}(",
            "    fp_t I_t,",
        ]
    )
    for sv in equations:
        lines.append(f"    fp_t &{sv},")
    lines.extend(
        [
            "    bool &spike_out",
            ") {",
        ]
    )

    # HLS pragmas
    if hls_tool == "vitis":
        lines.extend(
            [
                "    #pragma HLS PIPELINE II=1",
                "    #pragma HLS INTERFACE ap_ctrl_none port=return",
                "    #pragma HLS INTERFACE ap_none port=I_t",
            ]
        )
        for sv in equations:
            lines.append(f"    #pragma HLS INTERFACE ap_none port={sv}")
        lines.append("    #pragma HLS INTERFACE ap_none port=spike_out")
    else:  # catapult
        lines.append("    // Catapult: pipeline directive applied at synthesis")

    lines.append("")

    # Equations
    for sv, expr in equations.items():
        # Simple translation: replace common patterns
        c_expr = expr
        lines.append(f"    fp_t {sv}_next = (fp_t)({c_expr});")

    lines.append("")

    # Threshold / spike detection
    if equations:
        first_sv = list(equations.keys())[0]
        lines.extend(
            [
                "    // Threshold detection",
                "    const fp_t V_THRESH = (fp_t)(1.0);  // Configurable",
                f"    spike_out = ({first_sv}_next > V_THRESH);",
                "",
            ]
        )

    # Update state
    for sv in equations:
        lines.append(f"    {sv} = {sv}_next;")

    lines.extend(
        [
            "}",
            "",
            f"#endif // {guard}_HLS_H",
        ]
    )

    return "\n".join(lines)
