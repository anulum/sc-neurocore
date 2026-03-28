# Tutorial 63: Hardware Fault Resilience Testing

Test how your SNN degrades under hardware faults *before* deployment.
The resilience suite injects stuck-at faults, bit flips, dead synapses,
and stochastic computing biases, measures accuracy degradation per
fault rate, and identifies the most vulnerable layer.

## Why Resilience Testing

FPGA and ASIC hardware develops faults over time: radiation-induced
bit flips (space, medical), wear-out (automotive), manufacturing
defects. An SNN that drops from 97% to 50% accuracy with 1% stuck
weights is not deployable. One that drops to 95% is.

## Quick Start

```python
import numpy as np
from sc_neurocore.resilience import FaultResilienceSuite
from sc_neurocore.resilience.fault_suite import FaultType

rng = np.random.default_rng(42)

# Your evaluation function (returns accuracy)
def my_eval(weights):
    # Replace with actual inference + accuracy measurement
    # Returns float in [0, 1]
    return 0.95 - np.mean([np.abs(w).mean() for w in weights]) * 0.1

model_weights = [
    rng.standard_normal((128, 784)).astype(np.float32) * 0.05,
    rng.standard_normal((10, 128)).astype(np.float32) * 0.1,
]

suite = FaultResilienceSuite(eval_fn=my_eval, weights=model_weights)

# Sweep stuck-at-zero faults at increasing rates
report = suite.sweep(FaultType.STUCK_AT_ZERO, rates=[0.01, 0.05, 0.1, 0.2])
print(report.summary())
# Fault: STUCK_AT_ZERO
# Rate 0.01: accuracy 94.8% (Δ -0.2%)
# Rate 0.05: accuracy 93.1% (Δ -1.9%)
# Rate 0.10: accuracy 89.7% (Δ -5.3%)
# Rate 0.20: accuracy 81.2% (Δ -13.8%)
# Critical threshold: ~8% fault rate for >5% accuracy drop
```

## Full Audit

Test all fault types across all layers:

```python
full = suite.full_audit()
print(f"Most vulnerable layer: {full.most_vulnerable_layer()}")
print(f"Most damaging fault: {full.most_damaging_fault()}")
print(f"Overall resilience score: {full.resilience_score()}/100")

# Per-layer, per-fault breakdown
for layer, faults in full.layer_results.items():
    for fault_type, result in faults.items():
        print(f"  {layer} × {fault_type}: Δ={result.accuracy_drop:.1%} at 5% rate")
```

## Fault Types

| Type | Effect | SC-Specific? | Severity |
|------|--------|:------------:|----------|
| `STUCK_AT_ZERO` | Weights clamped to 0 | No | High (silences synapses) |
| `STUCK_AT_ONE` | Weights clamped to 1 | No | High (saturates activity) |
| `WEIGHT_BIT_FLIP` | Random bit flipped in Q8.8 | No | Medium (depends on bit position) |
| `DEAD_SYNAPSE` | Entire connections zeroed | No | High (structural damage) |
| `NOISY_MEMBRANE` | Gaussian noise on membrane | No | Low (SNNs are noise-tolerant) |
| `BITSTREAM_BIAS` | SC probability bias toward 0.5 | **Yes** | Medium (degrades SC precision) |

The `BITSTREAM_BIAS` fault is unique to stochastic computing — it
models LFSR correlation that causes bitstream probabilities to drift
toward 0.5. No other framework tests for this.

## Hardening Strategies

If resilience testing reveals vulnerability, SC-NeuroCore provides
mitigation:

| Strategy | How | Accuracy Cost |
|----------|-----|:------------:|
| Mismatch-aware training (Tutorial 48) | Inject faults during training | <1% |
| Weight redundancy | Duplicate critical synapses | 0% (more resources) |
| Error-correcting codes | SECDED on weight BRAMs | 0% (adds parity bits) |
| Threshold homeostasis (Tutorial 68) | Auto-regulate after faults | ~1% |

## Comparison

| Feature | SC-NeuroCore | Lava | snnTorch |
|---------|:-----------:|:----:|:--------:|
| Fault injection | 6 types | No | No |
| Per-layer vulnerability | Yes | No | No |
| SC-specific faults | Yes | No | No |
| Automated sweep | Yes | No | No |
| Resilience scoring | Yes | No | No |

## References

- Schuman et al. (2022). "Resilience and Robustness of Spiking Neural
  Networks for Neuromorphic Systems." IJCNN 2022.
- El-Sayed et al. (2018). "Spiking neural network robustness to
  permanent hardware faults." Neural Computing and Applications.
