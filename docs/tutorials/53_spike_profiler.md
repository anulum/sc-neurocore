# Tutorial 53: Spike-Level Training Profiler

Diagnose SNN training problems automatically.

## The Problem

SNN training fails silently. Dead neurons, gradient collapse, saturated layers
— all produce the same symptom: loss doesn't decrease. No existing framework
tells you *why*.

## Quick Start

```python
from sc_neurocore.profiling import SpikeProfiler
import numpy as np

profiler = SpikeProfiler()

# During training, record each layer's spikes and voltages
for epoch in range(100):
    for batch in dataloader:
        # ... forward pass ...
        profiler.record_step("hidden", spikes_hidden, voltages=v_hidden)
        profiler.record_step("output", spikes_output, voltages=v_output)

# Get diagnostic report
report = profiler.report()
print(report.summary())
```

## What It Detects

| Pathology | Severity | Trigger | Fix |
|-----------|----------|---------|-----|
| Dead neurons | CRITICAL if >50%, WARNING if >10% | Firing rate < 0.01 | Lower threshold, add noise |
| Saturated neurons | WARNING if >30% | Firing rate > 0.95 | Raise threshold, reduce input |
| Silent network | CRITICAL | Max rate < 0.001 | Check input encoding, lower all thresholds |
| Voltage collapse | WARNING | Voltage std < 1e-6 | Increase input current |
| Gradient explosion | CRITICAL | Max/mean norm > 100x | Clip gradients, reduce LR |
| Gradient vanishing | CRITICAL | First/last layer ratio > 100x | Skip connections, adaptive surrogate slope |

## Recording Gradients

```python
# Record surrogate gradient magnitudes for gradient health check
profiler.record_step(
    "hidden",
    spikes_hidden,
    voltages=v_hidden,
    gradients=surrogate_grad_hidden,
)
```

## Checking for Problems

```python
report = profiler.report()

if report.has_critical:
    print("CRITICAL issues found!")
    for p in report.pathologies:
        print(f"  [{p.severity.value}] {p.category} @ {p.layer}")
        print(f"    {p.message}")
        print(f"    Fix: {p.suggestion}")
```

## Per-Layer Statistics

```python
for name, stats in report.layer_stats.items():
    print(f"{name}:")
    print(f"  Neurons: {stats.n_neurons}")
    print(f"  Firing rate: {stats.firing_rates.mean():.3f}")
    print(f"  Dead: {stats.dead_neuron_count} ({stats.dead_neuron_fraction:.0%})")
    print(f"  Voltage: {stats.voltage_mean:.3f} +/- {stats.voltage_std:.3f}")
```
