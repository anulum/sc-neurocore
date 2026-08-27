# Hyper-Dimensional Computing — Binary Vector Algebra

High-dimensional binary vector algebra for symbolic reasoning in spiking networks. HDC maps naturally to stochastic computing hardware: bind = XOR gate, bundle = popcount tree, similarity = Hamming distance.

## Theory

HDC represents symbols as random binary vectors of dimension D (typically D >= 10,000). At high D, random vectors are quasi-orthogonal with high probability: `E[d_H(a,b)] = D/2`. Three operations form an algebra:

| Operation | Implementation | Property |
|-----------|---------------|----------|
| **Bind** (⊗) | XOR | Self-inverse: `a ⊗ a = 0`, `a ⊗ b ⊗ b = a` |
| **Bundle** (⊕) | Majority vote | Preserves similarity to all inputs |
| **Permute** (ρ) | Cyclic shift | Breaks commutativity for ordered structures |

## Components

- **`HDCEncoder`** — Generate random D-dimensional binary vectors and perform algebraic operations.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `dim` | 10000 | Hypervector dimension |
| `seed` | None | Seeds the encoder's own generator; a seeded encoder is fully deterministic for the same call order |
| `tie_policy` | "zeros" | Even-count bundle ties: `"zeros"` clears tied bits (historical strict majority), `"ones"` sets them, `"random"` decides them from a fresh seeded tie-break vector |

Methods: `generate_random_vector()`, `item(name)` (cached named item memory),
`bind(v1, v2)`, `bundle(vectors)`, `majority(sum_vec, count)` (shared bundle
kernel), `permute(v, shifts)`, `level_vectors(low, high, levels)` and
`encode_level(value, low, high, levels=16)` (linear level encoding whose
Hamming distance grows linearly with level separation, for scalar features).

- **`AssociativeMemory`** — Clean-up memory via Hamming distance nearest-neighbor lookup. Store labeled vectors, retrieve by similarity. Tolerates up to ~35% bit noise.

- **`CentroidHDClassifier`** — Nearest-centroid classifier over binary
hypervectors with mistake-driven retraining. Each class keeps a bipolar
accumulator; the centroid is its sign with exact zeros resolved by the
encoder's tie policy. `fit(vectors, labels)` accumulates, `predict(vector)`
returns the nearest centroid by Hamming distance, and
`retrain(vectors, labels, epochs)` applies the standard mistake-driven update
(add the misclassified example to its true class, subtract it from the
predicted class), returning the misclassification count per epoch.
Deterministic for a seeded encoder. Rejects non-binary or wrongly shaped
vectors, unknown retrain labels, and non-positive epochs with typed
`ValueError`s; the whole surface is enforced at 100% statement and branch
coverage by the hosted `HDC exact coverage` lane.

## Usage

```python
from sc_neurocore.hdc import HDCEncoder, AssociativeMemory
import numpy as np

np.random.seed(42)
enc = HDCEncoder(dim=10000)

# Create symbols
country = enc.generate_random_vector()
capital = enc.generate_random_vector()
usa = enc.generate_random_vector()
washington = enc.generate_random_vector()

# Encode: USA_record = bind(country, usa) ⊕ bind(capital, washington)
record = enc.bundle([
    enc.bind(country, usa),
    enc.bind(capital, washington),
])

# Query: "What is the capital of USA?" → bind(record, capital)
query = enc.bind(record, capital)

# Store in associative memory and retrieve
mem = AssociativeMemory()
mem.store("washington", washington)
mem.store("usa", usa)
print(mem.query(query))  # → "washington"
```

See [Tutorial 4: Hyper-Dimensional Computing](../tutorials/04_hyperdimensional_computing.md).

::: sc_neurocore.hdc.base
    options:
      show_root_heading: true
