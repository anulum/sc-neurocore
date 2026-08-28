<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — retained clipped rational-recovery map -->

# SC clipped rational-recovery map

`SCClippedRationalRecoveryMapNeuron` preserves the project recurrence formerly
carried by the Courbage-labelled class. It is a count-neutral SC identity with
no whole-model publication attribution.

- Source module: `sc_neurocore.neurons.models.sc_clipped_rational_recovery_map`
- Family: project-defined two-state map
- State: fast coordinate `x`, recovery coordinate `y`
- Event: upward crossing of `x_threshold`; no reset

## Recurrence equation

$$
f(x)=
\begin{cases}
\alpha x, & x<0,\\
\dfrac{\alpha x}{1+\alpha x}, & x\geq0,
\end{cases}
$$

$$
\begin{aligned}
x_{n+1} &= \operatorname{clip}(f(x_n)+y_n+I_n+j,-B,B),\\
y_{n+1} &= \operatorname{clip}(y_n-\beta(x_n+1),-B,B).
\end{aligned}
$$

Both candidates use the pre-step state and commit simultaneously. Rejected
non-finite inputs and candidates leave the public state unchanged.

## Parameters and defaults

| Field | Default | Meaning |
| --- | ---: | --- |
| `x` | `0.0` | initial fast coordinate |
| `y` | `0.0` | initial recovery coordinate |
| `alpha` | `3.0` | piecewise fast-map gain |
| `beta` | `0.001` | recovery decrement coefficient |
| `j` | `0.1` | constant fast-map offset |
| `x_threshold` | `1.0` | upward-crossing event threshold |
| `clip_bound` (`B`) | `1,000,000` | symmetric state saturation bound |

```python
from sc_neurocore.neurons.models.sc_clipped_rational_recovery_map import (
    SCClippedRationalRecoveryMapNeuron,
)

neuron = SCClippedRationalRecoveryMapNeuron()
trace, events = neuron.simulate(512, current=0.0, backend="auto")
```

`simulate()` accepts `python`, `rust`, `julia`, `go`, `mojo`, or `auto`.
All five explicit runtimes reproduce the complete binary64 trace and event
count bit-for-bit at the enrolled workloads.

## Benchmark evidence

`benchmarks/results/bench_sc_clipped_rational_recovery_map.json` records five
repeats of 2,000,000 retained-default iterations on a non-isolated loaded
workstation. All runtimes record zero events and zero full-trace state error.
Median times were 1406.11 ms (Python), 24.34 ms (Rust), 25.37 ms (Julia),
27.98 ms (Go), and 19.19 ms (Mojo). These are local-regression measurements,
not isolated-core release claims.

```bash
PYTHONPATH=src .venv/bin/python \
  benchmarks/bench_sc_clipped_rational_recovery_map.py \
  --json benchmarks/results/bench_sc_clipped_rational_recovery_map.json
```

## Verification and silicon evidence

The paired TOML and JSON schemas match the hand class for the complete
512-step retained-default trajectory. Q16.16 RTL preserves the 128-step event
vector with `x` and `y` errors below `0.0009` and `0.00046`; Q32.32 preserves
the same event vector with errors below `5e-8` and `3e-8`.

The generated Q32.32 module synthesises in Yosys to 35 coarse cells, including
the retained rational division. A depth-4 SymbiYosys/Z3 job proves the bounded
reset/event port-safety assertions. This is H2 coarse-synthesis evidence, not
timing closure, PPA, board deployment, or physical silicon.

The independent 512-step project receipt records the complete recurrence,
initial state, final/minimum/maximum/mean state features, and x-trace SHA-256.
Its test reimplements the recurrence without calling the production class or
schema expressions.

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q \
  tests/test_sc_clipped_rational_recovery_map_backends.py \
  tests/test_reference_sc_clipped_rational_recovery_map.py \
  tests/test_cosim_sc_clipped_rational_recovery_map.py

cd hdl/formal/catalogue
sby -f sc_clipped_rational_recovery_map.sby
```

See the [reference-trace contract](../../validation/reference_traces.md) and
[model fidelity status](../model_fidelity_status.md) for the catalogue-wide
evidence rules.
