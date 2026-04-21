# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Explicit RTL source emitters for LFSR-16 and Sobol-16

# Stochastic Source Emitters

SC-NeuroCore now exposes explicit standalone RTL emitters for the two
canonical 16-bit stochastic sources already used elsewhere in software:

- `Lfsr16Emitter` for the maximal-length `x^16 + x^14 + x^13 + x^11 + 1`
  source
- `Sobol16Emitter` for the 1D 16-bit Sobol source

These emitters do **not** silently replace any existing HDL flow. They are
standalone building blocks that can be instantiated where bit-exact source
generation is needed.

## Why these emitters exist

Before this addition the codebase had:

- software `Lfsr16` and `SobolGenerator` implementations
- tests for software and Rust parity
- no direct standalone RTL emitter modules under `sc_neurocore.hdl_gen`

That gap made the RTL path less explicit than the software path. The new
emitters close that gap without rewiring the existing top-level generator.

## Semantics

Both emitters use **compare-before-advance** semantics:

1. `bit_out` compares the current source state against `threshold`
2. the internal source state advances on the next clock edge

This matches the software and Rust encoder semantics already tested in the
repository.

## Python usage

```python
from sc_neurocore.hdl_gen import Lfsr16Emitter, Sobol16Emitter, VerilogGenerator

lfsr_rtl = Lfsr16Emitter(seed=0xACE1).generate()
sobol_rtl = Sobol16Emitter(seed=0x0042).generate()

generator = VerilogGenerator()
inline_lfsr = generator.emit_lfsr16_source()
inline_sobol = generator.emit_sobol16_source()
```

## Emitted module interface

Both standalone modules expose:

- `clk`
- `rst_n`
- `threshold[15:0]`
- `bit_out`
- source state registers for inspection

The LFSR module exports `state[15:0]`.  
The Sobol module exports `value[15:0]` and `index[15:0]`.

## Intended use

Use these emitters when you need:

- standalone RTL source blocks for FPGA or co-simulation
- explicit parity testing between software, Rust, and Verilog
- deterministic seed control for stochastic source generation

Do **not** treat them as proof that every HDL path in the repository is now
automatically sourced from these modules. They are explicit building blocks,
not an implicit global rewiring.
