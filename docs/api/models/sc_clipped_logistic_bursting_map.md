<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — retained clipped-logistic bursting recurrence -->

# SC clipped-logistic bursting map

`SCClippedLogisticBurstingMapNeuron` preserves the historical two-state
clipped-logistic fast/slow recurrence that was formerly exposed under the
Cazelles identity. It is count-neutral and intentionally carries no
whole-model publication attribution.

## Recurrence equation

$$
\begin{aligned}
x_{n+1} &= \operatorname{clip}(a x_n(1-x_n)-y_n+I_n,-2,2), \\
y_{n+1} &= y_n + \varepsilon(x_n-\sigma).
\end{aligned}
$$

A level event is emitted whenever the committed `x` is at or above
`x_threshold`; events do not reset state.

## Parameters and defaults

| Field | Default | Role |
| --- | ---: | --- |
| `x` | `0.1` | fast state |
| `y` | `0.0` | slow state |
| `a` | `3.8` | clipped-logistic coefficient |
| `epsilon` | `0.01` | slow update coefficient |
| `sigma` | `0.5` | slow set point |
| `x_threshold` | `0.9` | level-event threshold |

```python
from sc_neurocore.neurons.models.sc_clipped_logistic_bursting_map import (
    SCClippedLogisticBurstingMapNeuron,
)

neuron = SCClippedLogisticBurstingMapNeuron()
trace, events = neuron.simulate(2_000, current=0.05, backend="auto")
```

Python, Rust, Julia, and Go are binary64-exact at the enrolled points. Mojo is
validated per step with an ULP bound because fused multiply-add rounding is
amplified by the chaotic recurrence; the enrolled event counts remain stable.

## Benchmark

The committed benchmark evidence is
`benchmarks/results/bench_sc_clipped_logistic_bursting_map.json`. It records
the pinned two-million-step Python, Rust, Julia, Go, and Mojo measurements and
their parity results; it is implementation evidence, not a physical-silicon
performance claim.

The paired TOML/JSON schemas, project reference receipt, five-runtime tests,
Q16.16 bounded co-simulation, Q8.8 Yosys synthesis report, and depth-4
SymbiYosys/Z3 job all belong to this retained identity. The `I=0.05` Q16.16
long-window trajectory is explicitly excluded because fixed-point rounding is
chaotically amplified.

This retained identity does not add another source model to the public model
count. The source-faithful scalar model is documented at
[Cazelles map](cazelles_map.md).
