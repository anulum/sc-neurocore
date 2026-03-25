# Tutorial 84: Few-Shot Meta-Learning

Learn from 1-5 examples using spike-timing plasticity, not gradients.
Two approaches: Hebbian Associative Memory (HAAM) and Spike Prototypical Networks.

## HebbianFewShot (HAAM)

```python
import numpy as np
from sc_neurocore.few_shot import HebbianFewShot

learner = HebbianFewShot(n_features=128, n_classes=5, lr_hebbian=0.1)
rng = np.random.RandomState(42)

# 5-way, 1-shot
for c in range(5):
    pattern = rng.rand(128) * (c + 1) / 5
    learner.store(pattern, label=c)

query = rng.rand(128) * 3 / 5
predicted = learner.query(query)
print(f"Predicted: {predicted}")
```

## Few-Shot Episode

```python
support_x = [rng.rand(128) for _ in range(10)]
support_y = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
query_x = [rng.rand(128) for _ in range(5)]

predictions = learner.few_shot_episode(support_x, support_y, query_x)
```

## SpikePrototypeNet

Nearest-prototype classification in spike domain:

```python
from sc_neurocore.few_shot import SpikePrototypeNet

proto_net = SpikePrototypeNet(n_features=128, metric="cosine")
predictions = proto_net.classify(support_x, support_y, query_x)
```

| Method | Mechanism | Hardware |
|--------|-----------|----------|
| `HebbianFewShot` | Hebbian weight update | On-chip STDP |
| `SpikePrototypeNet` | Nearest prototype | Hamming distance |

Both accept spike-rate vectors or raw spike trains (auto-averaged).

Reference: HAAM (BICS 2024)

## API Reference

::: sc_neurocore.few_shot.haam
    options:
      show_root_heading: true
