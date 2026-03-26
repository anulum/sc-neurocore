# Bio — Biological Computing Substrates

Biological computing interfaces: DNA-based weight storage, gene regulatory network modulation, and neuromodulatory dynamics (dopamine, serotonin, norepinephrine).

## DNAEncoder — DNA Data Storage

Maps bitstreams to nucleotide sequences and back. Encoding: pairs of bits → nucleotides (00→A, 01→C, 10→G, 11→T). Decoding includes a configurable mutation rate that simulates sequencing errors.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `mutation_rate` | 0.001 | Per-nucleotide mutation probability during decode |

Odd-length bitstreams are zero-padded to even length.

## GeneticRegulatoryLayer — Gene Expression Modulation

Neural activity drives protein production; protein levels modulate neuron thresholds. Implements a first-order ODE: `dP/dt = α * spikes - β * P`, clipped to [0, 10].

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_neurons` | (required) | Number of neurons |
| `production_rate` | 0.01 | Protein production rate (α) |
| `decay_rate` | 0.005 | Protein decay rate (β) |

`get_threshold_modulators()` returns current protein levels — higher protein → higher effective threshold (inhibitory feedback).

## NeuromodulatorSystem — Global Emotional System

Three neuromodulators with environmental feedback:

| Chemical | Baseline | Effect |
|----------|----------|--------|
| Dopamine (DA) | 0.5 | Lowers threshold (excitation) |
| Serotonin (5-HT) | 0.5 | Reduces noise (stabilization) |
| Norepinephrine (NE) | 0.1 | Increases noise + gain (exploration) |

`update_levels(reward, stress)` adjusts chemicals. `modulate_neuron(params)` returns modified parameters.

## Usage

```python
from sc_neurocore.bio import DNAEncoder, GeneticRegulatoryLayer, NeuromodulatorSystem
import numpy as np

# DNA storage roundtrip
enc = DNAEncoder(mutation_rate=0.0)
bits = np.array([1, 0, 0, 1, 1, 1, 0, 0], dtype=np.uint8)
dna = enc.encode(bits)   # "GCTA"
recovered = enc.decode(dna)
assert np.array_equal(bits, recovered)

# Gene regulation
grn = GeneticRegulatoryLayer(n_neurons=100)
for _ in range(50):
    spikes = (np.random.rand(100) < 0.3).astype(float)
    grn.step(spikes)
thresholds = grn.get_threshold_modulators()

# Neuromodulation
nm = NeuromodulatorSystem()
nm.update_levels(reward=0.8, stress=0.2)
params = nm.modulate_neuron({"v_threshold": 1.0, "noise_std": 0.1})
```

::: sc_neurocore.bio.dna_storage
    options:
      show_root_heading: true

::: sc_neurocore.bio.grn
    options:
      show_root_heading: true

::: sc_neurocore.bio.neuromodulation
    options:
      show_root_heading: true
