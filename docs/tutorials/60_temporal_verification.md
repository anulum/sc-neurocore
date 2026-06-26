# Tutorial 60: Temporal Logic Verification

Prove safety properties of your SNN — or find counterexamples. Testing
checks N inputs; verification checks ALL inputs within bounds. No other
SNN framework provides formal temporal verification.

## Why Verification

For safety-critical SNN deployments (medical devices, autonomous
vehicles, industrial control), "it works on the test set" isn't enough.
You need *guarantees*:

- "Output neuron 5 ALWAYS responds within 10ms of stimulus"
- "Excitatory and inhibitory outputs NEVER fire simultaneously"
- "No neuron ever exceeds 30% firing rate"

The EU AI Act (Articles 50/52) requires such guarantees for high-risk
AI systems. SC-NeuroCore's verification module provides them.

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
spikes[105, 5] = 1  # neuron 5 fires 5 steps after stimulus at t=100
spikes[205, 5] = 0  # (would be checked if data is longer)

result = fires_within(
    spikes, neuron_id=5,
    stimulus_times=[100, 150],
    max_latency=10,
)

print(result.summary())
# [PASS] fires_within: Neuron 5 responds within 10 steps for all 2 stimuli
# Worst case: 5 steps (stimulus at t=100, response at t=105)
```

## Safety: No Simultaneous Conflicting Outputs

For classification networks, two class outputs should never fire
simultaneously:

```python
result = mutual_exclusion(spikes, neuron_set=[0, 1, 2, 3])

if result.result.value == "violated":
    print(f"VIOLATION: Neurons {result.counterexample.neurons} "
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
    print(f"Measured rate: {result.counterexample.rate:.3f}")
```

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

SC-NeuroCore uses bounded model checking — formal within the specified
input bounds and time horizon. For small networks (<100 neurons), this
provides complete coverage.

## Integration with FPGA Formal Verification

SC-NeuroCore also provides 18 SymbiYosys proof jobs and 130 formal statements
(100 assert, 7 assume, 23 cover) for RTL-level verification of generated
Verilog.
The temporal properties here verify the *algorithm*; SymbiYosys verifies
the *hardware implementation*.

## References

- Clarke et al. (2018). "Handbook of Model Checking." Springer.
- Huang et al. (2022). "Formal Verification of Neural Networks: A
  Survey." IEEE Access 10:125603-125618.
