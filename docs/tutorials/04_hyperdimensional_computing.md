<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Hyper-Dimensional Computing for Symbolic AI

HDC (Hyper-Dimensional Computing), also called Vector Symbolic Architecture
(VSA), represents symbols as random vectors in a *d*-dimensional space
(*d* = 10 000 typical). At this dimensionality, independently sampled binary
vectors are nearly orthogonal with high probability: expected Hamming distance
= *d*/2, standard deviation = *d*^(1/2)/2. Three algebraic operations --
binding, bundling, permutation -- suffice for structured symbolic reasoning.

**References**: Kanerva, "Hyperdimensional Computing: An Introduction to
Computing in Distributed Representation with High-Dimensional Random Vectors",
*Cognitive Computation* 1(2), 2009. Neubert et al., "An Introduction to
Hyperdimensional Computing for Robotics", *KI - Kunstliche Intelligenz* 33(4),
2019.

**Prerequisites**: Python 3.10+, `pip install sc-neurocore` (pure Python path;
the optional Rust engine adds SIMD acceleration but is not required).

## 1. The algebra

| Operation | Binary impl. | Symbol | Property |
|-----------|-------------|--------|----------|
| **Bind** | XOR | `a ^ b` | Self-inverse: `(a ^ b) ^ b = a` |
| **Bundle** | Majority vote | `[a, b, c] -> threshold(sum)` | Preserves similarity to constituents |
| **Permute** | Cyclic shift | `roll(a, k)` | Produces quasi-orthogonal vector for each *k* |

Binding associates two concepts (e.g. role-filler). Bundling superimposes
multiple items into a set. Permutation encodes positional/sequential structure.

## 2. Core API

```python
from sc_neurocore.hdc import HDCEncoder, AssociativeMemory

enc = HDCEncoder(dim=10_000)
```

`HDCEncoder` operates on binary `{0, 1}` NumPy vectors (dtype `uint8`).

```python
a = enc.generate_random_vector()  # shape (10000,), dtype uint8
b = enc.generate_random_vector()

bound = enc.bind(a, b)            # XOR
bundled = enc.bundle([a, b])      # majority vote
shifted = enc.permute(a, shifts=1)  # cyclic left shift
```

`AssociativeMemory` stores labelled prototype vectors and retrieves the
nearest match by Hamming distance:

```python
mem = AssociativeMemory()
mem.store("cat", cat_vec)
mem.store("dog", dog_vec)

label = mem.query(noisy_cat_vec)  # -> "cat"
```

## 3. Symbolic key-value memory

Encode country-capital pairs using the Multiply-Add-Permute (MAP) scheme:

```
record = bind(role_country, country) XOR-bundled-with bind(role_capital, capital)
```

```python
import numpy as np
from sc_neurocore.hdc import HDCEncoder, AssociativeMemory

D = 10_000
enc = HDCEncoder(dim=D)
rng = np.random.default_rng(42)

def make_vec():
    return rng.integers(0, 2, D, dtype=np.uint8)

# Atomic symbols
role_country, role_capital = make_vec(), make_vec()
atoms = {
    "France": make_vec(), "Germany": make_vec(), "Japan": make_vec(),
    "Paris": make_vec(),  "Berlin": make_vec(),  "Tokyo": make_vec(),
}

# Encode records: record_i = bind(role_country, country_i) + bind(role_capital, capital_i)
pairs = [("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo")]
records = []
for country, capital in pairs:
    bound_country = enc.bind(role_country, atoms[country])
    bound_capital = enc.bind(role_capital, atoms[capital])
    records.append(enc.bundle([bound_country, bound_capital]))

# Superimpose all records into a single memory vector
memory = enc.bundle(records)
```

## 4. Querying: "Capital of France?"

Unbinding reverses a bind. Since XOR is self-inverse, bind the query role
and the known key, then XOR with memory:

```python
# Unbind France, then unbind role_capital
probe = enc.bind(memory, atoms["France"])
answer = enc.bind(probe, role_capital)

# Find closest atom by Hamming distance
codebook = AssociativeMemory()
for name, vec in atoms.items():
    codebook.store(name, vec)

result = codebook.query(answer)
print(f"Capital of France -> {result}")  # Paris
```

The answer vector is noisy (bundling and multiple XOR operations degrade the
signal), but at *d* = 10 000 the correct atom remains the nearest neighbour.

## 5. Similarity metric

Hamming similarity = 1 - (Hamming distance / *d*). Random vectors score ~0.50.
Bound or permuted vectors also score ~0.50 (quasi-orthogonal). A bundled
vector scores > 0.50 against each constituent, with the excess depending on
the number of items bundled.

```python
def hamming_similarity(a, b):
    return 1.0 - np.count_nonzero(a ^ b) / len(a)

a, b = make_vec(), make_vec()
print(f"Random pair:    {hamming_similarity(a, b):.3f}")     # ~0.500
print(f"Self:           {hamming_similarity(a, a):.3f}")     # 1.000
print(f"Bound (a^b):    {hamming_similarity(a, enc.bind(a, b)):.3f}")  # ~0.500
print(f"Permuted:       {hamming_similarity(a, enc.permute(a)):.3f}")  # ~0.500
bundled = enc.bundle([a, b])
print(f"Bundle vs a:    {hamming_similarity(a, bundled):.3f}")  # ~0.65-0.75
```

## 6. Capacity limits

Bundle capacity scales as *O*(*d* / log *d*) (Kanerva 2009, Section 4). For
*d* = 10 000, retrieval degrades around 50-100 bundled items. Increasing *d*
extends capacity proportionally.

```python
codebook = AssociativeMemory()
vectors = [make_vec() for _ in range(200)]
for i, v in enumerate(vectors):
    codebook.store(str(i), v)

for n in [5, 10, 25, 50, 100]:
    bundled = enc.bundle(vectors[:n])
    nearest = codebook.query(bundled)
    is_constituent = int(nearest) < n
    print(f"N={n:3d}: nearest is constituent = {is_constituent}")
```

## 7. Sequence encoding with permutation

Permutation encodes order. For a sequence `[A, B, C]`:

```python
seq = enc.bundle([
    enc.permute(atoms["France"], shifts=0),   # position 0
    enc.permute(atoms["Germany"], shifts=1),   # position 1
    enc.permute(atoms["Japan"], shifts=2),     # position 2
])

# Retrieve position 1: inverse-permute, then query codebook
probe_pos1 = enc.permute(seq, shifts=-1)
print(codebook.query(probe_pos1))  # Germany (highest similarity)
```

Shift by -*k* inverts shift by *k*; the result is closest to the item at
position *k*.

## 8. Rust-accelerated path

When the compiled engine is available (`maturin develop --release` in
`engine/`), `sc_neurocore_engine.HDCVector` wraps the Rust `BitStreamTensor`
with SIMD popcount, XOR, bundle, and rotation. The Python API mirrors the
pure-NumPy version:

```python
from sc_neurocore_engine import HDCVector

a = HDCVector(10_000, seed=42)
b = HDCVector(10_000, seed=99)

bound = a * b                    # XOR bind (operator overload)
bundled = HDCVector.bundle([a, b])  # majority vote
shifted = a.permute(1)           # cyclic rotation
sim = a.similarity(b)            # 1.0 - normalised Hamming distance
```

The Rust path is 10-50x faster for *d* >= 10 000, relevant for large
codebooks or real-time inference on neuromorphic hardware pipelines.
