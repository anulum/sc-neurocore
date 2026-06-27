# Tutorial 84: Few-Shot Meta-Learning with Spikes

Learn from 1-5 examples using spike-timing plasticity, not gradient
descent. Two approaches: Hebbian Associative Memory (HAAM) for one-shot
pattern storage, and Spike Prototypical Networks for metric-based
few-shot classification.

## Why Few-Shot for SNNs

Standard SNN training requires thousands of examples and hundreds of
epochs. Few-shot learning stores new patterns in 1-5 presentations —
mimicking biological rapid learning (hippocampal one-shot encoding).

| Method | Examples Needed | Training Time | Hardware |
|--------|:--------------:|:------------:|:--------:|
| Standard backprop | 10,000+ | Hours | GPU |
| Few-shot (HAAM) | 1-5 | Milliseconds | On-chip STDP |
| Few-shot (prototypes) | 1-5 | Milliseconds | Hamming distance |

## HebbianFewShot (HAAM)

Hebbian Associative Memory stores patterns via one-shot weight updates:

```python
import numpy as np
from sc_neurocore.few_shot import HebbianFewShot

rng = np.random.RandomState(42)

learner = HebbianFewShot(
    n_features=128,
    n_classes=5,
    lr_hebbian=0.1,
)

# 5-way, 1-shot: store one example per class
for c in range(5):
    pattern = rng.rand(128).astype(np.float32) * (c + 1) / 5
    learner.store(pattern, label=c)
    print(f"Stored class {c}: mean={pattern.mean():.3f}")

# Query with a noisy version of class 2
query = rng.rand(128).astype(np.float32) * 3 / 5 + rng.randn(128).astype(np.float32) * 0.05
predicted = learner.query(query)
print(f"\nQuery predicted: class {predicted}")

# Bounded cosine scores, one per class
scores = learner.query_scores(query)
for c, s in enumerate(scores):
    print(f"  Class {c}: {s:.4f}")

# Defensive copy for hardware export or audit logging
weights = learner.export_weights()
```

### How HAAM Works

Storage: Hebbian outer product stores input-output association.
```
W += lr * (label_vector ⊗ input_pattern)
```

Retrieval: Matrix-vector multiply + winner-take-all.
```
scores = cosine(W[class], query_pattern)
predicted = argmax(scores)
```

This runs on-chip as a single STDP update (store) and a single
forward pass (retrieve). No iterative optimisation needed.

## Few-Shot Episode (N-way K-shot)

Standard few-shot evaluation protocol:

```python
# 5-way 2-shot episode
support_x = [rng.rand(128).astype(np.float32) for _ in range(10)]
support_y = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]  # 2 per class
query_x = [rng.rand(128).astype(np.float32) for _ in range(5)]

predictions = learner.few_shot_episode(support_x, support_y, query_x)
print(f"Predictions: {predictions}")
```

## SpikePrototypeNet

Nearest-prototype classification in spike-rate space. Compute class
prototypes from support examples, classify queries by distance:

```python
from sc_neurocore.few_shot import SpikePrototypeNet

proto_net = SpikePrototypeNet(
    n_features=128,
    metric="cosine",  # or "euclidean", "hamming"
)

predictions = proto_net.classify(support_x, support_y, query_x)
print(f"Prototype predictions: {predictions}")

# Prototype analysis
prototypes = proto_net.export_prototypes()
for c, p in prototypes.items():
    print(f"Class {c} prototype: mean={p.mean():.3f}, sparsity={np.mean(p < 0.1):.1%}")
```

### Distance Metrics

| Metric | Best For | Hardware |
|--------|----------|---------|
| Cosine | Spike-rate vectors | MAC + normalize |
| Euclidean | Continuous representations | Subtractor + MAC |
| Hamming | Binary spike patterns | XOR + popcount |

Hamming distance is ideal for neuromorphic hardware — XOR gates are
the cheapest operation on FPGA. In the Python API, hamming mode thresholds
vectors at zero and scores by negative normalized bit disagreement, so higher
scores still mean closer prototypes.

## Comparison

| Method | Mechanism | Memory | Speed | Hardware |
|--------|-----------|--------|-------|----------|
| HebbianFewShot | Hebbian outer product | O(N × K) | 1 update | On-chip STDP |
| SpikePrototypeNet | Nearest prototype | O(K × D) | 1 forward | Hamming distance |
| MAML (gradient) | Meta-gradient descent | O(N × K) | 5-10 inner steps | GPU only |

The HAAM and prototype paths have deterministic Python contracts plus Rust,
Julia, and Mojo validation mirrors for their vector episode kernels. MAML-style
gradient baselines are listed only as context; this tutorial does not implement
or benchmark a MAML inner loop.

## On-Chip Deployment

```python
# Export HAAM for FPGA
weights = learner.export_weights()
# The weight matrix stores all patterns — one MAC per query
# On iCE40: 128×5 = 640 weights × Q8.8 = 1.25 KB BRAM

# Export prototypes for FPGA
protos = proto_net.export_prototypes()
# Hamming distance: XOR + popcount, ~2 LUTs per feature bit
```

## Local Benchmark Evidence

The benchmark record at `benchmarks/results/bench_few_shot_haam.json` is a
local, non-isolated regression artifact. The current run recorded:

| Surface | Result |
|---------|--------|
| Python `HebbianFewShot.few_shot_episode` | 1000 calls, 3581.011 calls/s |
| Python `SpikePrototypeNet.classify` | 1000 calls, 4770.742 calls/s |
| Rust safety mirror | Compile passed; 5 unit tests passed |
| Julia mirror | `validate_haam()` and hamming prototype classification passed |
| Mojo kernel | Vector, temporal, and metric validation passed |

These numbers are suitable for local regression comparison only. They are not
isolated-core production performance claims.

## References

- Ororbia et al. (2024). "HAAM: Hebbian Associative Attention Memory
  for Few-Shot Learning." BICS 2024.
- Snell et al. (2017). "Prototypical Networks for Few-Shot Learning."
  NeurIPS 2017.
- Scherr et al. (2020). "One-Shot Learning with Spiking Neural Networks."
  bioRxiv 2020.01.24.
