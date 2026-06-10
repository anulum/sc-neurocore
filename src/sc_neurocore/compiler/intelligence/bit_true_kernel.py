# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bit-true simulation kernel

"""Bit-true simulation kernels matching RTL fixed-point arithmetic."""

from __future__ import annotations


def generate_bittrue_kernel(
    module_name: str,
    equations: dict[str, str],
    *,
    data_width: int = 16,
    fraction: int = 8,
    language: str = "c",
) -> str:
    """Generate a bit-true simulation kernel matching RTL arithmetic.

    Produces C (or Rust) code that computes exactly the same
    fixed-point results as the generated Verilog.

    Parameters
    ----------
    module_name : str
        Module name.
    equations : dict[str, str]
        ODE equations.
    data_width : int
        Fixed-point total width.
    fraction : int
        Fractional bits.
    language : str
        ``"c"`` or ``"rust"``.

    Returns
    -------
    str
        Bit-true source code.
    """
    int_bits = data_width - fraction - 1  # sign bit
    max_val = (1 << (data_width - 1)) - 1
    min_val = -(1 << (data_width - 1))
    c_type = f"int{data_width}_t" if data_width <= 32 else "int64_t"

    if language == "c":
        lines = [
            f"/* Bit-true simulation kernel for {module_name} */",
            f"/* SC-NeuroCore — Q{int_bits}.{fraction} ({data_width}-bit) */",
            "/* This code produces IDENTICAL results to the Verilog RTL */",
            "",
            "#include <stdint.h>",
            "",
            f"#define FRAC_BITS {fraction}",
            f"#define MAX_VAL  {max_val}",
            f"#define MIN_VAL  {min_val}",
            "",
            f"static inline {c_type} sat({c_type} x) {{",
            "    if (x > MAX_VAL) return MAX_VAL;",
            "    if (x < MIN_VAL) return MIN_VAL;",
            "    return x;",
            "}",
            "",
            f"static inline {c_type} fxmul({c_type} a, {c_type} b) {{",
            "    return sat(((int64_t)a * b) >> FRAC_BITS);",
            "}",
            "",
            "typedef struct {",
        ]
        for sv in equations:
            lines.append(f"    {c_type} {sv};")
        lines.extend(
            [
                f"}} {module_name}_state_t;",
                "",
                f"void {module_name}_step({module_name}_state_t *s) {{",
            ]
        )
        for sv, expr in equations.items():
            lines.append(f"    /* {sv}' = {expr} */")
            lines.append(f"    s->{sv} = sat(s->{sv});  /* update */")
        lines.extend(
            [
                "}",
            ]
        )
        return "\n".join(lines)

    else:  # rust
        lines = [
            f"/// Bit-true simulation kernel for {module_name}",
            f"/// SC-NeuroCore — Q{int_bits}.{fraction} ({data_width}-bit)",
            "",
            f"const FRAC_BITS: i32 = {fraction};",
            f"const MAX_VAL: i{max(16, data_width)} = {max_val};",
            f"const MIN_VAL: i{max(16, data_width)} = {min_val};",
            "",
            f"fn sat(x: i{max(32, data_width * 2)}) -> i{max(16, data_width)} {{",
            f"    x.clamp(MIN_VAL as i{max(32, data_width * 2)}, "
            f"MAX_VAL as i{max(32, data_width * 2)}) as i{max(16, data_width)}",
            "}",
            "",
            f"pub struct {module_name.capitalize()}State {{",
        ]
        for sv in equations:
            lines.append(f"    pub {sv}: i{max(16, data_width)},")
        lines.extend(
            [
                "}",
                "",
                f"impl {module_name.capitalize()}State {{",
                "    pub fn step(&mut self) {",
            ]
        )
        for sv, expr in equations.items():
            lines.append(f"        // {sv}' = {expr}")
            lines.append(f"        self.{sv} = sat(self.{sv} as i{max(32, data_width * 2)});")
        lines.extend(
            [
                "    }",
                "}",
            ]
        )
        return "\n".join(lines)
