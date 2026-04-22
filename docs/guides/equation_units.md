<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Opt-in dimensional checking for EquationNeuron -->

# Equation Units

`EquationNeuron` and `from_equations(...)` now support an opt-in strict
dimensional validation path.

## Modes

### `units="none"`

- default
- current behaviour
- raw numeric floats
- no dimensional validation

### `units="strict"`

- requires `pint`
- validates every equation before runtime compilation
- keeps the default path untouched

## What strict mode requires

When `units="strict"` is enabled:

- every `params` entry must be a `pint.Quantity`
- every `init` or `state` entry must be a `pint.Quantity`
- every dimensional `constants` entry must be a `pint.Quantity`
- `dt` must be a time `Quantity`
- if the equation references the special input `I`, you must provide
  `input_unit`

## Important boundary

Bare numbers inside equation strings are dimensionless. That means
dimensional thresholds and resets should be written through named quantity
constants, not inline numeric literals.

Recommended:

```python
threshold="v > v_threshold"
reset="v = v_reset"
constants={
    "v_threshold": -50.0 * ureg.millivolt,
    "v_reset": -65.0 * ureg.millivolt,
}
```

Not recommended in strict mode:

```python
threshold="v > -50"
reset="v = -65"
```

## Example

```python
import pint
from sc_neurocore.neurons.equation_builder import from_equations

ureg = pint.UnitRegistry()

neuron = from_equations(
    "dv/dt = (-(v - E_L) + R * I) / tau_m",
    threshold="v > v_threshold",
    reset="v = v_reset",
    params={
        "E_L": -65.0 * ureg.millivolt,
        "R": 100e6 * ureg.ohm,
        "tau_m": 10.0 * ureg.millisecond,
    },
    init={"v": -65.0 * ureg.millivolt},
    constants={
        "v_threshold": -50.0 * ureg.millivolt,
        "v_reset": -65.0 * ureg.millivolt,
    },
    dt=0.1 * ureg.millisecond,
    units="strict",
    input_unit=1.0 * ureg.nanoampere,
)
```

## Runtime behaviour

Strict mode validates dimensions up front, then runs the solver on a
numerically coherent internal representation. `get_state()` returns
quantities in the original display units that were provided during
construction.

## FPGA export

The strict-unit route also works with the equation compiler.
`compile_to_verilog(...)` receives the validated numeric internal
representation, while named dimensional constants such as
`v_threshold` and `v_reset` still export as normal Verilog parameters.

`equation_to_fpga(...)` accepts the same strict-mode arguments as
`from_equations(...)`:

- `constants`
- `units`
- `input_unit`

## Installation

```bash
pip install sc-neurocore[units]
```
