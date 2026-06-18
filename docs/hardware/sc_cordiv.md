<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — CORDIV state-machine contract -->

# CORDIV State-Machine Contract

This page documents the CORDIV contract implemented for
`sc_neurocore.utils.bitstreams.sc_divide` and aligned hardware
`hdl/sc_cordiv.v`.

## Scope

This is stochastic-computing division evidence. It is a contract and correctness
description, not a throughput or hardware-performance claim.

## State-machine contract

- Inputs must be same-shape numerator and denominator bitstreams.
  The denominator path is required to have matching shape and length.
- Output is a binary bitstream (values in `{0, 1}`).
- Reset initialises the hardware output register to `0`.
- Per-bit transition:
  - `x[t] = 1` -> `z[t] = 1`
  - `x[t] = 0` and `y[t] = 1` -> `z[t] = 0`
  - `x[t] = 0` and `y[t] = 0` -> `z[t] = z[t-1]` (hold previous)

The stochastic estimate converges toward `P(x=1) / P(y=1)` when
`P(x=1) <= P(y=1)`. Correlated streams and short stream lengths can bias the
estimate, so this page documents the circuit contract rather than a throughput
or accuracy claim.

## Hardware Mapping

`hdl/sc_cordiv.v` implements the contract with one clocked output register:

| Condition | Next `z` |
|-----------|----------|
| `rst` | `0` |
| `x == 1` | `1` |
| `x == 0 && y == 1` | `0` |
| `x == 0 && y == 0` | hold previous `z` |

The HDL path has no divider macro. It is a sequential stochastic divider: one
output bit is emitted per input bit pair, and the previous quotient bit carries
the hold state.

## Cited implementation and tests

- `src/sc_neurocore/utils/bitstreams.py`
- `hdl/sc_cordiv.v`
- `tests/test_cordiv_division.py`
- `tests/test_sc_division.py`
- `tests/test_adaptive_length.py`
- `tests/test_sc_convergence.py`

## Focused Python example

```python
from sc_neurocore.utils import RNG
from sc_neurocore.utils.bitstreams import (
    bitstream_to_probability,
    generate_bernoulli_bitstream,
    sc_divide,
)

numerator = generate_bernoulli_bitstream(0.6, 1024, rng=RNG(7))
denominator = generate_bernoulli_bitstream(0.9, 1024, rng=RNG(11))
quotient = sc_divide(numerator, denominator)
print(bitstream_to_probability(quotient))
```

For deterministic low-discrepancy streams, choose lengths returned by
`adaptive_length(...)`; it rounds up to a power of two for Sobol compatibility.

## Replication commands

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_cordiv_division.py \
  tests/test_sc_division.py \
  tests/test_sc_convergence.py \
  tests/test_adaptive_length.py \
  tests/test_chaotic_encoder.py \
  -q
```
