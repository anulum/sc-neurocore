<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 46: NeuroBench Benchmarking

Generate NeuroBench-compatible evaluation reports for standardized
comparison with other neuromorphic frameworks.

## 1. Compute Metrics

```python
import numpy as np
from sc_neurocore.benchmarks import compute_metrics

result = compute_metrics(
    predictions=np.array([0, 1, 2, 0, 1]),
    targets=np.array([0, 1, 2, 0, 0]),
    spike_counts=np.array([30, 25, 40, 20, 35]),
    weights=[np.random.randn(10, 8)],
    timesteps=16,
    task="mnist",
)
print(result.summary())
```

## 2. Export JSON

```python
print(result.to_neurobench_json())
```

## 3. Available Tasks

```python
from sc_neurocore.benchmarks import TASKS
for name, task in TASKS.items():
    print(f"{name}: {task.n_classes} classes, baseline {task.baseline_accuracy}")
```

Tasks: keyword spotting, DVS gesture, heartbeat anomaly, MNIST, SHD.

## Further Reading

- [NeuroBench](https://neurobench.ai/)
- [API: Benchmarks](../api/benchmarks.md)
