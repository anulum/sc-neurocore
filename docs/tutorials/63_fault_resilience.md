# Tutorial 63: Hardware Fault Resilience Testing

Test how your SNN degrades under hardware faults.

## Quick Start

```python
from sc_neurocore.resilience import FaultResilienceSuite, FaultModel
from sc_neurocore.resilience.fault_suite import FaultType

suite = FaultResilienceSuite(eval_fn=my_eval, weights=model_weights)

# Sweep stuck-at-zero faults
report = suite.sweep(FaultType.STUCK_AT_ZERO, rates=[0.01, 0.05, 0.1, 0.2])
print(report.summary())

# Full audit: all fault types, per-layer
full = suite.full_audit()
print(f"Most vulnerable layer: {full.most_vulnerable_layer()}")
```

## Fault Types

| Type | Effect | SC-Specific? |
|------|--------|--------------|
| STUCK_AT_ZERO | Weights clamped to 0 | - |
| STUCK_AT_ONE | Weights clamped to 1 | - |
| WEIGHT_BIT_FLIP | Sign inversion | - |
| DEAD_SYNAPSE | Connections zeroed | - |
| NOISY_MEMBRANE | Gaussian noise on weights | - |
| BITSTREAM_BIAS | SC probability bias toward 0.5 | YES |
