# Fault Resilience — Hardware Fault Injection Suite

Systematic fault injection and degradation analysis for SNN deployments. Generates degradation curves (accuracy vs fault rate) and identifies the most vulnerable layers.

## Fault Models

| Fault Type | Description | SC Relevance |
|-----------|-------------|--------------|
| `STUCK_AT_ZERO` | Weight forced to 0 | Dead synapse in FPGA LUT |
| `STUCK_AT_ONE` | Weight forced to 1 | Stuck bit in BRAM |
| `WEIGHT_BIT_FLIP` | Sign flip of weight | SEU (single-event upset) from radiation |
| `DEAD_SYNAPSE` | Weight zeroed | Manufacturing defect |
| `NOISY_MEMBRANE` | Additive noise proportional to weight std | Thermal noise in analog circuits |
| `BITSTREAM_BIAS` | Probability shifted toward 0.5 | SC-specific: correlator degradation |

## Components

- **`FaultResilienceSuite`** — Main test harness.

| Parameter | Type | Meaning |
|-----------|------|---------|
| `eval_fn` | callable | `f(weights) → accuracy` — evaluation function |
| `weights` | list of ndarray | Baseline (unfaulted) weight matrices |

**Key methods:**

- `inject_fault(fault)` — Apply fault model, return faulted weight copies
- `run_single(fault)` → `FaultResult` — One injection experiment
- `sweep(fault_type, rates, per_layer)` → `ResilienceReport` — Sweep fault rates
- `full_audit()` → `ResilienceReport` — All fault types × all rates × all layers

- **`FaultModel`** — Configuration: fault_type, rate (0-1), optional layer_index, seed.
- **`FaultResult`** — Result: accuracy_before, accuracy_after, degradation.
- **`ResilienceReport`** — Collection of results with `degradation_curve()`, `most_vulnerable_layer()`, `summary()`.

## Usage

```python
from sc_neurocore.resilience import FaultResilienceSuite, FaultModel
from sc_neurocore.resilience.fault_suite import FaultType
import numpy as np

def eval_fn(weights):
    # Your model evaluation here
    return accuracy

weights = [np.random.randn(64, 32), np.random.randn(10, 64)]
suite = FaultResilienceSuite(eval_fn=eval_fn, weights=weights)

# Single fault experiment
result = suite.run_single(FaultModel(FaultType.STUCK_AT_ZERO, rate=0.1))
print(f"Degradation: {result.degradation:.3f}")

# Sweep fault rates
report = suite.sweep(FaultType.STUCK_AT_ZERO, rates=[0.01, 0.05, 0.1, 0.2, 0.5])
curve = report.degradation_curve(FaultType.STUCK_AT_ZERO)

# Full audit: all fault types × all layers
full = suite.full_audit()
print(f"Most vulnerable layer: {full.most_vulnerable_layer()}")
print(full.summary())
```

See [Tutorial 63: Fault Resilience](../tutorials/63_fault_resilience.md).

::: sc_neurocore.resilience
    options:
      show_root_heading: true
