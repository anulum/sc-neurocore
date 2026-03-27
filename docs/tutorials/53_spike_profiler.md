# Tutorial 53: Spike-Level Training Profiler

Diagnose SNN training problems automatically. The profiler monitors
spike rates, membrane voltages, and gradient magnitudes per layer,
detects 6 common pathologies, and suggests fixes — all without you
reading a single log file.

## The Problem

SNN training fails silently. Dead neurons, gradient collapse, saturated
layers — all produce the same symptom: loss doesn't decrease. No
existing framework tells you *why* training isn't working.

## Quick Start

```python
from sc_neurocore.profiling import SpikeProfiler
import numpy as np

profiler = SpikeProfiler()
rng = np.random.default_rng(42)

# During training, record each layer's spikes and voltages
for epoch in range(10):
    for batch in range(100):
        spikes_hidden = (rng.random((32, 128)) < 0.05).astype(np.float32)
        v_hidden = rng.standard_normal((32, 128)).astype(np.float32)
        spikes_output = (rng.random((32, 10)) < 0.3).astype(np.float32)
        v_output = rng.standard_normal((32, 10)).astype(np.float32)

        profiler.record_step("hidden", spikes_hidden, voltages=v_hidden)
        profiler.record_step("output", spikes_output, voltages=v_output)

# Get diagnostic report
report = profiler.report()
print(report.summary())
```

## What It Detects

| Pathology | Severity | Trigger | Suggested Fix |
|-----------|----------|---------|---------------|
| Dead neurons | CRITICAL if >50% | Firing rate < 0.01 | Lower threshold, add noise |
| Saturated neurons | WARNING if >30% | Firing rate > 0.95 | Raise threshold, reduce input |
| Silent network | CRITICAL | Max rate < 0.001 | Check input encoding, lower all thresholds |
| Voltage collapse | WARNING | Voltage std < 1e-6 | Increase input current |
| Gradient explosion | CRITICAL | Max/mean norm > 100× | Clip gradients, reduce LR |
| Gradient vanishing | CRITICAL | First/last layer ratio > 100× | Skip connections, adaptive surrogate |

## Recording Gradients

For gradient health monitoring, record surrogate gradient magnitudes:

```python
# After backward pass, record gradient statistics
profiler.record_step(
    "hidden",
    spikes_hidden,
    voltages=v_hidden,
    gradients=surrogate_grad_hidden,  # optional
)
```

## Checking for Problems

```python
report = profiler.report()

if report.has_critical:
    print("CRITICAL issues found:")
    for p in report.pathologies:
        if p.severity.value >= 2:  # CRITICAL
            print(f"  [{p.severity.name}] {p.category} @ {p.layer}")
            print(f"    {p.message}")
            print(f"    Fix: {p.suggestion}")

# Example output:
# CRITICAL issues found:
#   [CRITICAL] dead_neurons @ hidden
#     95 of 128 neurons (74.2%) have firing rate < 0.01
#     Fix: Lower threshold from 1.0 to ~0.5, or add input noise
```

## Per-Layer Statistics

```python
for name, stats in report.layer_stats.items():
    print(f"\n{name}:")
    print(f"  Neurons: {stats.n_neurons}")
    print(f"  Firing rate: {stats.firing_rates.mean():.4f} "
          f"(min={stats.firing_rates.min():.4f}, max={stats.firing_rates.max():.4f})")
    print(f"  Dead: {stats.dead_neuron_count} ({stats.dead_neuron_fraction:.0%})")
    print(f"  Saturated: {stats.saturated_neuron_count} ({stats.saturated_neuron_fraction:.0%})")
    print(f"  Voltage: {stats.voltage_mean:.3f} ± {stats.voltage_std:.3f}")
    if stats.gradient_norm is not None:
        print(f"  Gradient norm: {stats.gradient_norm:.6f}")
```

## Integration with Studio Training Monitor

The Studio's Training Monitor shows per-layer spike rates as horizontal
bars. The profiler provides the diagnostic layer underneath — detecting
whether those rates indicate healthy training or pathology.

```python
# In the Studio:
# 1. Start training in Training Monitor
# 2. Watch layer spike rate bars
# 3. If a layer goes to 0% (dead) or 100% (saturated):
#    - The profiler detects this automatically
#    - Suggested fix appears in the training log
```

## Comparison

| Feature | SC-NeuroCore Profiler | snnTorch | Norse | NEST |
|---------|:--------------------:|:--------:|:-----:|:----:|
| Automatic pathology detection | Yes | No | No | No |
| Per-layer spike statistics | Yes | Manual | Manual | Yes |
| Gradient health monitoring | Yes | Manual (PyTorch) | Manual | No |
| Suggested fixes | Yes | No | No | No |
| Dead neuron detection | Yes | No | No | No |

## References

- Zenke & Vogels (2021). "The Remarkable Robustness of Surrogate
  Gradient Learning." Neural Computation 33(4):899-925.
- Rathi & Roy (2021). "DIET-SNN: Direct Input Encoding and Leakage
  and Threshold Optimization." IEEE TNNLS.
