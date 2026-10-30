<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Standalone fixed-point function units -->

# Standalone fixed-point function units

Generate a single combinational function through the public compiler API:

```python
from pathlib import Path
from sc_neurocore.compiler import (
    emit_function_unit_verilog,
    function_unit_functions,
)

print(function_unit_functions("v2"))
source = emit_function_unit_verilog(
    "atan2", data_width=32, fraction=16, function_lowering="v2"
)
Path("function_unit.v").write_text(source, encoding="utf-8")
```

The module is named `function_unit`. It reuses the equation compiler's existing
function arithmetic and lookup tables. Its interface contains no neuron state,
integration step, clock or reset.

## Ports and fixed-point encoding

Every port is a signed `data_width`-bit word with `fraction` fractional bits.
An integer input code `n` represents `n / 2**fraction`; decode the result with
the same scale. Input codes span `-2**(data_width-1)` through
`2**(data_width-1)-1`.

| Port | Direction | Meaning |
| --- | --- | --- |
| `argument0` | input | First argument, or the sole argument of a unary function. |
| `argument1` | input | Independent second argument, present only for binary functions. |
| `result` | output | Encoded result using the selected lowering's rounding and saturation. |

For `atan2(y, x)`, `argument0` carries `y` and `argument1` carries `x`.
Binary arguments are separate ports; neither argument is tied to the other.
The returned source records the function, lowering, format and port contract
beside the module declaration.

The interface has zero cycles of sequential latency. Combinational propagation
delay, achievable clock frequency, area and power require synthesis or physical
measurements of the generated design.

## Function discovery and modes

`function_unit_functions("v2")` returns the sorted canonical functions supported
by the integer function-unit lowering. `function_unit_functions("lut-v1")`
returns the legacy table functions and the exact functions shared by both modes.
Use these returned inventories rather than assuming that a function exists in
both modes.

`emit_function_unit_verilog("expit", ...)` selects the canonical `sigmoid`
function. Discovery lists `sigmoid` once. Cube root uses the equation grammar's
power-one-third lowering. Random draws and inline arithmetic constructs such
as `rand`, `abs`, `clip` and `max` are not standalone catalogue entries and are
refused.

This API defaults to `function_lowering="v2"`. It does not change the legacy
default of the existing equation compiler. Specify `"lut-v1"` explicitly to
generate a legacy unit. Domain restrictions, table endpoints, range reduction,
rounding and saturation remain those of the selected equation-compiler
lowering; this wrapper introduces no new numerical approximation or accuracy
guarantee.

## Accepted formats and refusals

The word width must be an integer from 2 through 32. The fractional width must
be a non-negative integer smaller than the word width, at most 30, and the sum
of the word and fractional widths must not exceed 61. Boolean and floating-point
geometry values are refused rather than coerced.

Unknown functions, unavailable legacy functions, unsupported formats and
unknown lowerings raise `ValueError`. A non-string function name or non-integer
geometry raises `TypeError`. Function names are admitted against the canonical
inventory before an expression is parsed.

## Simulate a generated unit

For the default Q8.8 sine unit, input code `256` represents one radian:

```verilog
module tb;
  reg signed [15:0] argument0;
  wire signed [15:0] result;
  function_unit dut(.argument0(argument0), .result(result));
  initial begin
    argument0 = 16'sd256;
    #1;
    $display("%0d", $signed(result));
    $finish;
  end
endmodule
```

Save `emit_function_unit_verilog("sin")` as `function_unit.v` and the testbench
as `tb.v`, then run an installed Icarus Verilog toolchain:

```bash
iverilog -g2012 -s tb -o sim function_unit.v tb.v
vvp sim
```

The repository's owning tests execute the emitted RTL for both lowerings over
Q8.8, Q4.12, Q16.16, Q9.9, Q8.24 and Q1.15. They compare actual simulator output
with the existing integer reference at signed endpoints, domain boundaries,
independent binary operands and seeded interior inputs. These differential
checks also exhaust every signed input combination for selected two-, three-
and four-bit words, including LUT indices wider than their argument ports.
They establish the tested numerical behaviour; they are not a formal proof
over every possible input word or a measurement on a physical device.
