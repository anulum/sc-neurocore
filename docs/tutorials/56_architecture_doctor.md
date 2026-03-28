# Tutorial 56: SNN Architecture Doctor

Diagnose your SNN architecture automatically. Get severity-ranked
findings with specific fix recommendations — before training, after
training, and before FPGA deployment. The doctor checks hardware fit,
weight health, neuron activity, and coding efficiency in one call.

## Why an Architecture Doctor

SNN design has many failure modes that aren't caught by training loss:

| Problem | Symptom | Doctor Detection |
|---------|---------|-----------------|
| Network too big for FPGA | Synthesis fails | CRITICAL: hardware_fit |
| 90% of neurons never fire | Accuracy plateaus | CRITICAL: dead_neurons |
| Hidden layer 10× wider than needed | Wasted LUTs | WARNING: overprovisioned |
| Weights outside [0,1] SC range | Q8.8 clipping | INFO: weight_sc_range |

The doctor catches these before you waste hours training or synthesising.

## Quick Start

```python
from sc_neurocore.doctor import diagnose

report = diagnose(
    layer_sizes=[(784, 256), (256, 10)],
    target="ice40",
    bitstream_length=128,
)

print(report.summary())
print(f"\nHealth score: {report.score}/100")
# Health score: 72/100
# [WARNING] hardware_fit: 87% LUT utilization — close to limit
# [WARNING] architecture_bottleneck: 784→256 is 3x reduction (ok),
#   but 256→10 is 25x reduction (severe bottleneck)
# [INFO] coding_overprovisioned: L=128 for 10 output neurons is wasteful
```

## Full Diagnosis with Weights + Spike Rates

After training, provide weights and measured spike rates for deeper
analysis:

```python
import numpy as np

rng = np.random.default_rng(42)
layers = [(64, 32), (32, 10)]
weights = [
    rng.standard_normal((32, 64)).astype(np.float32) * 0.3,
    rng.standard_normal((10, 32)).astype(np.float32) * 0.3,
]
rates = [np.full(32, 0.15, dtype=np.float32),
         np.full(10, 0.1, dtype=np.float32)]

report = diagnose(
    layer_sizes=layers,
    weights=weights,
    spike_rates=rates,
    target="artix7",
    bitstream_length=256,
)

for f in report.findings:
    if f.severity.value != "ok":
        print(f"[{f.severity.value:>8s}] {f.category}: {f.message}")
        print(f"           Fix: {f.suggestion}")
```

## What It Checks

| Category | What | Severity |
|----------|------|----------|
| `hardware_fit` | LUT utilisation vs FPGA capacity | CRITICAL if >100% |
| `hardware_overprovisioned` | Network uses <10% of FPGA | INFO |
| `weight_sparsity` | >90% near-zero weights | WARNING |
| `weight_outliers` | Max/mean ratio > 10× | WARNING |
| `weight_sc_range` | Weights outside [0, 1] | INFO |
| `dead_neurons` | >50% silent (rate < 0.01) | CRITICAL |
| `saturated_neurons` | >30% always firing (rate > 0.95) | WARNING |
| `architecture_bottleneck` | >4× width reduction between layers | WARNING |
| `coding_overprovisioned` | Large L with few neurons | INFO |
| `coding_underprovisioned` | Small L with many neurons | WARNING |

## Health Score

0-100 scale. 100 = no issues found.

- Each CRITICAL finding: -10 points
- Each WARNING finding: -5 points
- Each INFO finding: -1 point

| Score | Meaning |
|-------|---------|
| 90-100 | Healthy — deploy with confidence |
| 70-89 | Minor issues — review warnings |
| 50-69 | Significant issues — fix before deployment |
| <50 | Critical issues — architecture needs redesign |

## Integration with Studio

The doctor runs automatically when you click **Pipeline** on the
Network Canvas. Validation errors include doctor findings alongside
graph structure checks.

## Comparison

No other SNN framework provides automated architecture diagnostics.
The closest equivalent is running Yosys and manually interpreting
resource reports — the doctor automates and extends this.

## References

- Sze et al. (2017). "Efficient Processing of Deep Neural Networks."
  Proceedings of the IEEE.
