<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# SCResettingWilsonHRNeuron

**Module:** `sc_neurocore.neurons.models.sc_resetting_wilson_hr`
**Rust engine:** `sc_neurocore_engine::neurons::SCResettingWilsonHRNeuron`

`SCResettingWilsonHRNeuron` preserves the historical SC-NeuroCore recurrence
that was formerly published under `WilsonHRNeuron`: unit membrane capacitance,
`r=0.1`, level detection at `v>=0.4`, and hard `v=-0.7` reset while preserving
the RK4 recovery candidate.

It is a project specialisation and is not attributed to Wilson's 1999
continuous `C=0.8` equations. Use
[`WilsonHRNeuron`](wilson_hr.md) for the literature model.

## Parameters and defaults

The historical parameter contract fixes unit capacitance and an initial state
of `v=-0.7`, `r=0.1`. The default integration step is `dt=0.05`; `v>=0.4`
detects an event, after which only voltage is reset to `-0.7`.

```python
from sc_neurocore.neurons.models.sc_resetting_wilson_hr import (
    SCResettingWilsonHRNeuron,
)

neuron = SCResettingWilsonHRNeuron()
trace, spikes = neuron.simulate(1_000, current=2.0, backend="auto")
```

## Verification

The 1,000-step compatibility anchor produces one reset event and retains the
pre-split binary64 trace digest. Python, production Rust/PyO3, independent
safety Rust, Julia, Go, and Mojo execute the same recurrence. Rust, Julia, and
Go are bit-identical to Python over the enrolled complete trajectories; Mojo is
bounded to `2.5e-12`, with exact reset-event counts. Native rejection is
failure-atomic at the public Python boundary.

Paired TOML/JSON schemas exercise the same historical recurrence. Generated
Q16.16 RTL has exact hand/schema/RTL event counts of 0, 1, and 4 over 5,000
steps at `I=0`, `I=2`, and `I=10`; the committed core also passes Yosys coarse
synthesis and a depth-4 public-port reset-safety proof. These are bounded
fixed-point claims, not timing, PPA, device, or universal real-number
equivalence claims. The source-bound five-runtime measurement is recorded in
`benchmarks/results/bench_sc_resetting_wilson_hr_simulate.json` as local,
non-isolated regression evidence only.
