# Few-Shot Meta-Learning — Hebbian Associative Memory

Learn from 1-5 examples using spike-timing plasticity instead of gradient descent. Two approaches: Hebbian weight storage and prototypical network classification.

## HebbianFewShot — Associative Memory

Support patterns stored via one-shot Hebbian update: `memory[label] += lr * pattern`. Queries classified by cosine similarity to stored memories. The `few_shot_episode()` method handles the full N-way K-shot protocol: reset → store support set → classify query set.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_features` | (required) | Input feature dimension |
| `n_classes` | (required) | Number of classes |
| `lr_hebbian` | 0.1 | Hebbian learning rate for storage |

Accepts spike-rate vectors `(n_features,)` or raw spike trains `(T, n_features)` — automatically averaged over time.

## SpikePrototypeNet — Prototypical Network

Computes class prototypes as mean spike-rate vectors from the support set. Classifies queries by nearest prototype using cosine or Euclidean distance. Stateless — no internal weights to maintain.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_features` | (required) | Feature dimension |
| `metric` | "cosine" | Distance metric: "cosine" or "euclidean" |

## Usage

```python
from sc_neurocore.few_shot import HebbianFewShot, SpikePrototypeNet
import numpy as np

# 5-way 1-shot with Hebbian memory
learner = HebbianFewShot(n_features=64, n_classes=5)
support_x = [np.random.rand(64) for _ in range(5)]
support_y = [0, 1, 2, 3, 4]
query_x = [np.random.rand(64) for _ in range(10)]
predictions = learner.few_shot_episode(support_x, support_y, query_x)

# Prototypical network (no training needed)
proto = SpikePrototypeNet(n_features=64, metric="cosine")
predictions = proto.classify(support_x, support_y, query_x)
```

**Reference:** HAAM (BICS 2024).

See [Tutorial 84: Few-Shot Meta-Learning](../tutorials/84_few_shot.md).

::: sc_neurocore.few_shot.haam
    options:
      show_root_heading: true
      members:
        - HebbianFewShot
        - SpikePrototypeNet
