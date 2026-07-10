<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
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
emitters close that gap while preserving the existing top-level Dense-layer
wiring.

## Semantics

Both emitters use **software-parity** stream semantics:

1. reset initialises the exposed state to the first generated source sample
2. `bit_out` compares that current generated sample against `threshold`
3. the internal source state advances on the next clock edge

This matches the software and Rust encoder contract, where `Lfsr16.encode(...)`
and `SobolGenerator.encode(...)` advance before comparing each packed output
bit. The RTL still exposes the current sample before the next clocked advance,
so downstream testbenches can inspect the same value that drives `bit_out`.

## Python usage

```python
from sc_neurocore.hdl_gen import Lfsr16Emitter, Sobol16Emitter, VerilogGenerator

lfsr_rtl = Lfsr16Emitter(seed=0xACE1).generate()
sobol_rtl = Sobol16Emitter(seed=0x0042).generate()

generator = VerilogGenerator()
inline_lfsr = generator.emit_lfsr16_source()
inline_sobol = generator.emit_sobol16_source()

generator.add_layer("StochasticSource", "rng_lfsr", {"source_type": "LFSR", "seed": 0xBEEF})
generator.add_layer("StochasticSource", "rng_sobol", {"source_type": "Sobol", "seed": 0x0042})
top_and_sources_rtl = generator.generate()

source_rtl = generator.emit_sources_from_ir(
    {
        "nodes": [
            {
                "name": "rng_lfsr",
                "type": "StochasticSource",
                "params": {"source_type": "LFSR", "seed": 0xBEEF},
            },
            {
                "name": "rng_sobol",
                "type": "StochasticSource",
                "params": {"source_type": "Sobol", "seed": 0x0042},
            },
        ]
    }
)
```

`emit_sources_from_ir(...)` is also exported at package level for callers that
already hold an IR payload and do not need a `VerilogGenerator` instance.
Source module names are validated as HDL identifiers before emission. Explicit
source seeds must be integer values; invalid identifiers, duplicate module
names, unknown source kinds, and non-integer seeds are rejected instead of being
coerced into generated RTL.

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
- SC-NIR FPGA compile artefacts via `result.scnir_source_modules` and
  `result.scnir_source_manifest`, including stream `signal_kind` and
  recurrent stream `delay_steps`
- `compile-nir` output directories that pair source modules with the validated
  `scnir_document.json` used to generate them
- machine-readable compile evidence in `scnir_source_manifest.json`, including
  direct versus AER interconnect selection, graph size, and aggregate
  `scnir_signal_kinds` plus `scnir_signal_routes` for mixed analogue/spiking
  exports
- CLI-level co-simulation of emitted source modules selected from
  `scnir_source_manifest.json` across direct/Sobol, AER/LFSR, and
  recurrent/LFSR exported networks

Do **not** treat them as proof that every HDL path in the repository is now
automatically sourced from these modules. They are explicit building blocks,
not an implicit global rewiring. The CLI co-simulation matrix proves the
standalone source modules in those output directories follow the canonical
advance-before-compare contract; it is not yet full-network HDL co-simulation.
