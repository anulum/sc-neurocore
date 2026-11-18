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

CORDIV (T.-H. Chen and J. P. Hayes, "Design of division circuits for
stochastic computing", IEEE ISVLSI 2016, doi:10.1109/ISVLSI.2016.48) needs
**correlated** inputs. When `x`
and `y` are compared against one shared random sequence with
`P(x=1) <= P(y=1)`, every 1 of `x` falls on a 1 of `y`; `z` then holds the
value of `x` at the last bit where `y` was 1. If the sequence's successive
values are independent, the density of `z` converges to
`P(x=1 | y=1) = P(x=1) / P(y=1)`. In that regime the table above gives the
same output as CORDIV's multiplexer and flip-flop, which copy `x` when `y = 1`
and hold otherwise; the two differ only on `x = 1, y = 0`, which correlated
inputs never produce.

The successive values matter. A 16-bit LFSR read every clock, as
`hdl/sc_bitstream_encoder.v` is, does not qualify: each state is the previous
one shifted by one bit. With both streams compared against one such LFSR
(densities 0.3 and 0.6, seed `0x1234`, 32,768 bits), `z` equals `x` bit for
bit and its density is 0.303, not 0.502. The output equals `x` whenever
`2 P(x) <= P(y)`; at other densities it misses the quotient too (0.5 and 0.6
give 0.899 against 0.833, 0.4 and 0.5 give 0.580 against 0.803). Reading the
LFSR once every 16 steps gives 0.488 against a quotient of 0.499.

For **independent** streams the circuit converges to
`P(x) / (P(x) + P(y) - P(x) P(y))`, which is not the quotient: densities 0.3
and 0.6 give about 0.417 instead of 0.5. Short streams add sampling error on
top. This page documents the circuit contract and its operating condition, not
a throughput or accuracy claim.

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
- `tests/test_hdl_sc_cordiv_mux_add_cosim.py` (the RTL against `sc_divide`,
  bit for bit, on random, correlated and reset stimulus)
- `tests/test_cordiv_division_sc_divide.py`
- `tests/test_cordiv_division_adaptive_length.py`
- `tests/test_sc_division.py`
- `tests/test_adaptive_length.py`
- `tests/test_sc_convergence_cordiv_monotonicity.py`

## Focused Python example

```python
import numpy as np

from sc_neurocore.utils.bitstreams import bitstream_to_probability, sc_divide

shared = np.random.default_rng(7).random(4096)  # one random sequence for both streams
numerator = (shared < 0.6).astype(np.uint8)
denominator = (shared < 0.9).astype(np.uint8)
quotient = sc_divide(numerator, denominator)
print(bitstream_to_probability(quotient))  # close to 0.6 / 0.9
```

For deterministic low-discrepancy streams, choose lengths returned by
`adaptive_length(...)`; it rounds up to a power of two for Sobol compatibility.

## Replication commands

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_hdl_sc_cordiv_mux_add_cosim.py \
  tests/test_cordiv_division_sc_divide.py \
  tests/test_cordiv_division_adaptive_length.py \
  tests/test_sc_division.py \
  tests/test_sc_convergence_cordiv_monotonicity.py \
  tests/test_adaptive_length.py \
  tests/test_chaotic_encoder.py \
  -q
```
