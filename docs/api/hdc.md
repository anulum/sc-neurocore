# Hyper-Dimensional Computing

High-dimensional binary/bipolar vector algebra for symbolic reasoning in spiking networks.

- `HDCEncoder` — Generate random D-dimensional hypervectors, bind (XOR), bundle (majority vote), permute (cyclic roll). D typically >= 10,000 for noise tolerance.
- `AssociativeMemory` — Nearest-neighbor lookup via Hamming distance. Store labeled patterns, retrieve by similarity.

HDC maps naturally to SC hardware: bind = XOR gate, bundle = popcount tree, similarity = Hamming distance.

```python
from sc_neurocore.hdc import HDCEncoder, AssociativeMemory

enc = HDCEncoder(dim=10000)
v1 = enc.generate_random_vector()
v2 = enc.generate_random_vector()
bound = enc.bind(v1, v2)
bundled = enc.bundle([v1, v2])
```

See [Tutorial 4: Hyper-Dimensional Computing](../tutorials/04_hyperdimensional_computing.md).

::: sc_neurocore.hdc.base
    options:
      show_root_heading: true
