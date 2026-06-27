# Few-Shot Meta-Learning — Hebbian Associative Memory

Learn from 1-5 examples using spike-timing plasticity instead of gradient
descent. Two approaches are exposed: class-indexed Hebbian memory and
prototypical classification in spike-rate space.

## HebbianFewShot — Associative Memory

Support patterns are stored via one-shot Hebbian update:
`memory[label] += lr * pattern`. Queries are classified by cosine similarity to
the stored class memories. The `few_shot_episode()` method handles the full
N-way K-shot protocol: reset -> store support set -> classify query set.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_features` | (required) | Input feature dimension |
| `n_classes` | (required) | Number of classes |
| `lr_hebbian` | 0.1 | Hebbian learning rate for storage |

Accepts spike-rate vectors `(n_features,)` or raw spike trains `(T, n_features)` — automatically averaged over time.

`query_scores()` returns one bounded cosine score per class. Classes without
support examples score `0.0`, and querying before storage raises `ValueError`.
`export_weights()` returns a defensive copy of the class-memory matrix for
hardware export or downstream inspection.

## SpikePrototypeNet — Prototypical Network

Computes class prototypes as mean spike-rate vectors from the support set.
Classifies queries by nearest prototype using cosine similarity, negative
Euclidean distance, or negative normalized Hamming disagreement. It stores the
most recent prototypes only so they can be inspected or exported.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_features` | (required) | Feature dimension |
| `metric` | "cosine" | Distance metric: "cosine", "euclidean", or "hamming" |

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
prototypes = proto.export_prototypes()
```

**Reference:** HAAM (BICS 2024).

## Validation and Benchmark Evidence

The public Python API is covered by the module-specific
`tests/test_few_shot.py` surface. The maintained polyglot mirrors are:

| Surface | Scope | Local check |
|---------|-------|-------------|
| Python | Public API, vector and temporal inputs, exports, validation guards | `pytest tests/test_few_shot.py` |
| Rust | Safety mirror for vector HAAM and prototype episodes | `rustc --edition=2021 --test src/sc_neurocore/accel/rust/safety/haam.rs` |
| Julia | Vector HAAM and prototype validation mirror | `julia --startup-file=no --history-file=no ... validate_haam()` |
| Mojo | Standalone vector/temporal HAAM and prototype validation kernel | `mojo src/sc_neurocore/accel/mojo/kernels/haam.mojo` |

`benchmarks/results/bench_few_shot_haam.json` records the latest local,
non-isolated evidence. The Python public API measured 1000 deterministic calls
at 3581.011 Hebbian episodes/s and 4770.742 prototype classifications/s on the
current workstation. Rust, Julia, and Mojo validation checks all passed in that
same run. The artifact is a local regression record, not an isolated production
benchmark claim.

See [Tutorial 84: Few-Shot Meta-Learning](../tutorials/84_few_shot.md).

::: sc_neurocore.few_shot.haam
    options:
      show_root_heading: true
      members:
        - HebbianFewShot
        - SpikePrototypeNet
