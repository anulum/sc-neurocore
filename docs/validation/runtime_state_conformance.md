<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# How much of a model's state a foreign runtime can carry

A native lane returns arrays. Nothing about those arrays says which of the
model's declared state variables they are. The question this page answers, per
model and per lane, is: of the state this model declares, which variables could
that lane transport, which would it drop, and does it export anything the model
has no name for at all.

The answer is generated, not asserted:
`docs/_generated/runtime_state_conformance.json`
(`sc-neurocore.runtime-state-conformance.v1`), derived from the declared layouts
and the lane packets. A test fails when the file and the live contract disagree.

## The packet

Each lane declares what it transports as a
`sc-neurocore.runtime-state-packet.v1` packet:

| Field | Meaning |
|---|---|
| `runtime` | Lane identifier, for example `rust-batch` |
| `exports` | Declared-state variable names the lane can return, in the model's own vocabulary |
| `carries_initial_snapshot` | Whether it reports the state the run started from |
| `carries_parameters` | Whether it accepts parameter overrides |

Coverage against one model's declared state then splits into `carried`,
`dropped` and `unnameable` — the last being values the lane exports that this
model does not declare, and which therefore cannot be placed in a result under
those names.

## What the catalogue looks like today

The Rust batch lane transports one scalar trace, the soma voltage, from a model
constructed with its own defaults. Across all 185 catalogue models (10 of which
declare no state at all):

| | |
|---|---|
| Declared variables it could carry | 123 |
| Declared variables it would drop | 415 |
| Models fully accounted for | 18 |
| Models it can name nothing in | 52 |

Restricted to the 158 models the lane can currently run, of which 152 have a
declared layout: 111 carried, 356 dropped, and 41 in which it can name nothing.
`PinskyRinzelNeuron` is one of those 41 — it declares
`v_s, v_d, h, n, s, c, q, ca` and has no `v` — so the lane records no state for
it and says so in the result's custody notes rather than naming a trace after a
variable the model does not have.

The Go, Julia and Mojo lanes have no packet yet. The matrix says so by omission
rather than implying they transport everything.

## What this matrix does not say

It does not say whether a lane is built, installed, or able to run a given model
on this machine. That is a property of a toolchain, not of a contract, and a
matrix that mixed the two would change with the box it was generated on.
`tests/test_rust_python_neuron_parity.py` holds the binding coverage question.

It also does not say that a carried variable is *correct* — only that the lane
can name it. Numerical agreement between lanes is the parity and cosimulation
work, and it is reported on its own pages.

## Regenerating

```bash
python tools/runtime_state_conformance.py --write     # regenerate
python tools/runtime_state_conformance.py --check     # fail on drift
python tools/runtime_state_conformance.py --summary   # print the census
```

The file records no timestamp and no commit hash. It changes when a declared
layout or a lane packet changes, and at no other time.
