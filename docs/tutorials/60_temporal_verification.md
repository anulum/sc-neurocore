<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Formal verification inventory and qualification -->
# Tutorial 60: Temporal Logic Verification

Check temporal safety and liveness properties on a recorded spike raster,
and get a counterexample when one fails. Each check reads one bounded trace
and decides only what that trace shows: where the trace ends before a
question is settled, the answer is `unknown`. Guarantees over all inputs
come from the RTL-level formal proofs described at the end.

## Why Verification

For safety-critical SNN deployments (medical devices, autonomous
vehicles, industrial control), "it works on the test set" isn't enough.
You need precise statements such as:

- "Output neuron 5 ALWAYS responds within 10ms of stimulus"
- "Excitatory and inhibitory outputs NEVER fire simultaneously"
- "No neuron ever exceeds 30% firing rate"

The checks below state such properties precisely and test them on the
runs you record. A pass covers those runs, not every possible input.

## Available Properties

```python
from sc_neurocore.verification.temporal_properties import (
    fires_within,          # response time guarantee
    mutual_exclusion,      # no simultaneous conflicting outputs
    rate_bound,            # maximum firing rate bound
    refractory_guarantee,  # minimum inter-spike interval
    causal_order,          # A always fires before B
    bounded_activity,      # total spikes in window <= limit
)
```

## Response Time Guarantee

Prove that a neuron responds within a deadline after stimulus:

```python
import numpy as np

# Simulate spike data: 200 timesteps, 10 neurons
rng = np.random.default_rng(42)
spikes = (rng.random((200, 10)) < 0.05).astype(np.int8)
spikes[105, 5] = 1  # neuron 5 fires 5 steps after the stimulus at t=100
spikes[158, 5] = 1  # and 8 steps after the stimulus at t=150

result = fires_within(
    spikes, neuron_id=5,
    stimulus_times=[100, 150],
    max_latency=10,
)

print(result.summary())
# [PASS] fires_within: Neuron 5 responds within 10 steps for all 2 stimuli
```

A spike exactly `max_latency` steps after the stimulus still counts. A
stimulus whose deadline falls after the last step, with no spike yet,
gives `unknown` rather than a failure: the trace cannot show a later
response.

## Safety: No Simultaneous Conflicting Outputs

For classification networks, two class outputs should never fire
simultaneously:

```python
result = mutual_exclusion(spikes, neuron_set=[0, 1, 2, 3])

if result.result.value == "violated":
    print(f"VIOLATION: Neurons {result.counterexample.neuron_ids} "
          f"fired simultaneously at t={result.counterexample.timestep}")
else:
    print("PASS: No simultaneous firing in the output set")
```

## Rate Safety Bound

Ensure no neuron exceeds a maximum firing rate (prevents runaway
excitation):

```python
result = rate_bound(
    spikes, neuron_id=0,
    max_rate=0.3,       # max 30% firing rate
    window_size=20,     # measured over 20-step windows
)

if result.result.value == "violated":
    print(f"Rate exceeded at window starting t={result.counterexample.timestep}")
    print(result.counterexample.description)  # e.g. "Rate 0.350 > 0.300 in window [..]"
```

A trace shorter than one window is never reported as passing: it is
`violated` when the whole trace already exceeds the bound, and `unknown`
otherwise.

## Minimum Refractory Period

Verify that inter-spike intervals respect a minimum gap (hardware
constraint or biological requirement):

```python
result = refractory_guarantee(spikes, neuron_id=0, min_gap=5)
# Checks that no two spikes from neuron 0 are within 5 timesteps
```

## Causal Ordering

Prove that neuron A always fires before neuron B within a delay bound:

```python
result = causal_order(
    spikes, neuron_a=0, neuron_b=3,
    max_delay=10,
)
# Verifies: every spike of B is preceded by a spike of A within 10 steps
```

A spike of B in the first `max_delay` steps with no spike of A before it
gives `unknown`: its cause may lie before the recording started.

## Bounded Activity

Limit total spikes from a population within any time window (prevents
runaway avalanches):

```python
result = bounded_activity(
    spikes,
    neuron_set=[0, 1, 2, 3, 4],
    window_size=10,
    max_total_spikes=5,
)
# PASS means: at most 5 total spikes from these 5 neurons in any 10-step window
```

## Formal vs Empirical Verification

| Approach | Coverage | Speed | Guarantee |
|----------|---------|-------|-----------|
| Testing (random inputs) | Sampled | Fast | Probabilistic |
| Bounded model checking | All inputs within bounds | Moderate | Formal (bounded) |
| Exhaustive (small networks) | All possible states | Slow | Complete |

The checks in this tutorial are trace checks: they read one recorded
raster, so they belong to the first row. Bounded model checking over all
inputs is what the SymbiYosys proofs below do for the generated RTL.

## Integration with FPGA Formal Verification

SC-NeuroCore also provides 92 SymbiYosys proof jobs and 589 formal statements
(402 assert, 133 assume, 54 cover) for RTL-level verification of generated
Verilog.
The temporal properties here verify the *algorithm*; SymbiYosys verifies
the *hardware implementation*.

## References

- Clarke et al. (2018). "Handbook of Model Checking." Springer.
- Huang et al. (2022). "Formal Verification of Neural Networks: A
  Survey." IEEE Access 10:125603-125618. Neither Crossref nor OpenAlex
  holds a record of this entry; it is kept as written while it is under
  review. The next entry is a verified survey of the field.
- Huang, X., Kroening, D., Ruan, W., Sharp, J., Sun, Y., Thamo, E., Wu, M.,
  & Yi, X. (2020). "A survey of safety and trustworthiness of deep neural
  networks: Verification, testing, adversarial attack and defence, and
  interpretability." Computer Science Review 37:100270.
  doi:10.1016/j.cosrev.2020.100270
