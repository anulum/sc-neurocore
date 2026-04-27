<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Research-stage async AER HDL path -->

# Async AER HDL

SC-NeuroCore now includes a research-stage `async_aer` HDL emission path.

## Boundary

This is intentionally not a full asynchronous micropipeline backend.

Current behaviour:

- existing compute path remains clocked
- final spike vector is wrapped in a 4-phase AER-style request/acknowledge interface
- output address is derived from the first active spike bit

That makes this a safe additive scaffold for experimentation, not a
replacement for the stable synchronous HDL path.

## API

```python
from sc_neurocore.hdl_gen import VerilogGenerator

gen = VerilogGenerator(module_name="async_top")
gen.add_layer("Dense", "dense0", {"n_neurons": 4})
gen.add_layer("Dense", "dense1", {"n_neurons": 4})

rtl = gen.generate(mode="async_aer")
```

Or directly:

```python
rtl = gen.emit_async_aer()
```

## Generated interface

The async wrapper currently emits:

- `clk`
- `rst_n`
- `input_bus`
- `aer_ack`
- `aer_req`
- `aer_addr`
- `output_bus`

`output_bus` mirrors the wrapped spike vector so the experimental AER path
can still be compared against the existing synchronous output.

## Verification level

Current verification is limited to:

- structural assertions in Python tests
- HDL syntax smoke compile under `iverilog`

It is not yet:

- Verilator timing verification
- full QDI correctness proof
- sync-vs-AER behavioural parity harness on a real network workload

## Recommendation

Use this path only for research and comparison work. Keep production HDL on
the default synchronous emitter until a stronger parity and timing harness
exists.
