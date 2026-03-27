<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 46: NeuroBench Benchmarking

Generate standardised evaluation reports compatible with the
[NeuroBench](https://neurobench.ai/) framework for fair comparison
with other neuromorphic systems. Reports include accuracy, spike
efficiency, parameter count, and energy estimates.

## Why NeuroBench

Every framework reports different metrics in different ways. NeuroBench
provides a standard: same tasks, same metrics, same reporting format.
This makes cross-framework comparison honest and reproducible.

## Compute Metrics

```python
import numpy as np
from sc_neurocore.benchmarks import compute_metrics

rng = np.random.default_rng(42)

result = compute_metrics(
    predictions=np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]),
    targets=np.array([0, 1, 2, 0, 0, 2, 0, 1, 2, 1]),
    spike_counts=rng.integers(10, 50, size=10),
    weights=[rng.standard_normal((128, 64)).astype(np.float32)],
    timesteps=16,
    task="mnist",
)

print(result.summary())
# Task: mnist
# Accuracy: 80.0%
# Total spikes: 312
# Spikes per sample: 31.2
# Spike efficiency: 2.56% accuracy per spike
# Parameters: 8,192
# Compute ops: 131,072 (params × timesteps)
# Energy estimate: 0.013 mJ (at 45nm CMOS)
```

## Export NeuroBench JSON

The standard NeuroBench reporting format:

```python
json_report = result.to_neurobench_json()
print(json_report)
# {
#   "task": "mnist",
#   "accuracy": 0.80,
#   "n_parameters": 8192,
#   "n_timesteps": 16,
#   "total_spikes": 312,
#   "spikes_per_sample": 31.2,
#   "energy_mj": 0.013,
#   "framework": "sc-neurocore",
#   "version": "3.14.0"
# }

# Save for submission
from pathlib import Path
Path("neurobench_result.json").write_text(json_report)
```

## Available Tasks

NeuroBench defines standardised benchmarks:

```python
from sc_neurocore.benchmarks import TASKS

for name, task in TASKS.items():
    print(f"{name:20s} | {task.n_classes:>3d} classes | "
          f"baseline {task.baseline_accuracy:.1%} | {task.description}")
```

| Task | Classes | Baseline | Description |
|------|---------|----------|-------------|
| mnist | 10 | 99.5% (ANN) | Handwritten digits |
| shd | 20 | 85% (SNN) | Spiking Heidelberg Digits (speech) |
| dvs_gesture | 11 | 95% (SNN) | DVS128 Gesture recognition |
| heartbeat | 2 | 90% (ANN) | ECG anomaly detection |
| keyword | 12 | 96% (ANN) | Speech keyword spotting |

## Full Benchmark Pipeline

```python
from sc_neurocore.training import SpikingNet, train_epoch, evaluate, auto_device
from sc_neurocore.benchmarks import compute_metrics
from sc_neurocore.training.utils import SpikeMonitor

# 1. Train model
device = auto_device()
model = SpikingNet(n_input=784, n_hidden=128, n_output=10).to(device)
# ... training loop ...

# 2. Evaluate with spike counting
monitor = SpikeMonitor(model)
test_loss, test_acc = evaluate(model, test_loader, n_timesteps=25, device=device)
total_spikes = sum(
    monitor.get(name).sum().item() for name in monitor.layer_names
    if monitor.get(name) is not None
)

# 3. Generate NeuroBench report
result = compute_metrics(
    predictions=all_predictions,
    targets=all_targets,
    spike_counts=spike_counts_per_sample,
    weights=[p.detach().cpu().numpy() for p in model.parameters()],
    timesteps=25,
    task="mnist",
)

print(result.summary())
print(result.to_neurobench_json())
```

## SC-NeuroCore vs NeuroBench Leaderboard

Honest comparison (measured, not estimated):

| Task | SC-NeuroCore | Best Published | Gap |
|------|-------------|---------------|-----|
| MNIST | 99.49% (ConvSNN) | 99.72% (SEW-ResNet) | -0.23% |
| SHD | not yet measured | 95.1% (SpikFormer) | — |
| DVS Gesture | not yet measured | 98.2% (TET) | — |

We publish measured numbers only. SHD and DVS Gesture benchmarks are
pending — we'll update this table when results are available.

## References

- Yik et al. (2024). "NeuroBench: Advancing Neuromorphic Computing
  through Collaborative, Fair and Representative Benchmarking."
  Nature Communications.
- Cramer et al. (2020). "The Heidelberg Spiking Data Sets for the
  Systematic Evaluation of Spiking Neural Networks." IEEE TNNLS.
