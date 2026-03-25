# Spike-Based Few-Shot Meta-Learning

Learn from 1-5 examples using spike-timing plasticity instead of gradient descent.

- `HebbianFewShot` — Associative memory with one-shot Hebbian storage. Support patterns stored via `lr * pattern` weight update, queries classified by cosine similarity to stored memories. Supports `few_shot_episode()` for N-way K-shot evaluation. (HAAM, BICS 2024)
- `SpikePrototypeNet` — Prototypical network in spike domain. Computes class prototypes as mean spike-rate vectors, classifies queries by nearest prototype (cosine or Euclidean).

Both accept spike-rate vectors or raw spike trains (T, n_features) — automatically averaged.

```python
from sc_neurocore.few_shot import HebbianFewShot, SpikePrototypeNet
```

See [Tutorial 84: Few-Shot Meta-Learning](../tutorials/84_few_shot.md) for usage examples.

::: sc_neurocore.few_shot.haam
    options:
      show_root_heading: true
      members:
        - HebbianFewShot
        - SpikePrototypeNet
