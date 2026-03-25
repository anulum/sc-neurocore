# Tutorial 56: SNN Architecture Doctor

Diagnose your SNN architecture automatically. Get severity-ranked
findings with specific fix recommendations.

## Quick Start

```python
from sc_neurocore.doctor import diagnose

report = diagnose(
    layer_sizes=[(784, 256), (256, 10)],
    target="ice40",
    bitstream_length=128,
)
print(report.summary())
print(f"Health score: {report.score}/100")
```

## Full Diagnosis with Weights + Spike Rates

```python
import numpy as np

layers = [(64, 32), (32, 10)]
weights = [np.random.randn(32, 64) * 0.3, np.random.randn(10, 32) * 0.3]
rates = [np.full(32, 0.15), np.full(10, 0.1)]

report = diagnose(
    layer_sizes=layers,
    weights=weights,
    spike_rates=rates,
    target="artix7",
    bitstream_length=256,
)

for f in report.findings:
    if f.severity.value != "ok":
        print(f"[{f.severity.value}] {f.category}: {f.message}")
        print(f"  Fix: {f.suggestion}")
```

## What It Checks

| Category | What | Severity |
|----------|------|----------|
| hardware_fit | LUT utilization vs FPGA capacity | CRITICAL if >100% |
| hardware_overprovisioned | Network too small for target | INFO |
| weight_sparsity | >90% near-zero weights | WARNING |
| weight_outliers | Max/mean ratio > 10x | WARNING |
| weight_sc_range | Weights outside [0, 1] | INFO |
| dead_neurons | >50% silent (rate < 0.01) | CRITICAL |
| saturated_neurons | >30% always firing | WARNING |
| architecture_bottleneck | >4x width reduction between layers | WARNING |
| coding_overprovisioned | Large L with few neurons | INFO |
| coding_underprovisioned | Small L with many neurons | WARNING |

## Health Score

0-100 scale: 100 = no issues. Each CRITICAL finding deducts 10 points,
WARNING deducts 5, INFO deducts 1.
