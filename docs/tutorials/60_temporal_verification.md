# Tutorial 60: Temporal Logic Verification

Prove safety properties of your SNN — or find counterexamples.

## Why Verification?

Testing checks N inputs. Verification checks ALL inputs (within bounds).
EU AI Act requires verification for high-risk AI. No SNN framework has this.

## Available Properties

```python
from sc_neurocore.verification.temporal_properties import (
    fires_within, mutual_exclusion, rate_bound,
    refractory_guarantee, causal_order, bounded_activity,
)
```

## Response Time Guarantee

```python
result = fires_within(spikes, neuron_id=5, stimulus_times=[100, 200], max_latency=10)
print(result.summary())
# [PASS] fires_within: Neuron 5 responds within 10 steps for all 2 stimuli
```

## Safety: No Simultaneous Outputs

```python
result = mutual_exclusion(spikes, neuron_set=[0, 1, 2, 3])
if result.result.value == "violated":
    print(f"Collision at t={result.counterexample.timestep}")
```

## Rate Safety Bound

```python
result = rate_bound(spikes, neuron_id=0, max_rate=0.3, window_size=20)
```

## Minimum Refractory Period

```python
result = refractory_guarantee(spikes, neuron_id=0, min_gap=5)
```

## Causal Ordering

```python
result = causal_order(spikes, neuron_a=0, neuron_b=3, max_delay=10)
# Verifies A always fires before B within 10 steps
```

## Activity Bound

```python
result = bounded_activity(spikes, neuron_set=[0, 1, 2], window_size=10, max_total_spikes=5)
```
