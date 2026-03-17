# Identity Substrate

Persistent spiking neural network for identity continuity.

**Module**: `sc_neurocore.identity`

## Architecture

Three-population network with STDP-driven small-world connectivity:

| Population | Neuron Model | Size (default) | Role |
|-----------|-------------|:--------------:|------|
| cortical | HodgkinHuxley | 500 | Main processing, sensory input |
| inhibitory | WangBuzsaki | 200 | Fast-spiking balance/stability |
| memory | HindmarshRose | 100 | Bursting pattern storage via attractors |

Six projections: E->E (small-world STDP), E->I, I->E, E->M, M->E, I->I.

## Classes

### IdentitySubstrate

```python
from sc_neurocore.identity import IdentitySubstrate

substrate = IdentitySubstrate(n_cortical=500, n_inhibitory=200, n_memory=100)
substrate.inject_experience("The proof relies on Cauchy's integral formula.")
substrate.run(duration=1.0)
state = substrate.extract_state()
health = substrate.health_check()
```

### TraceEncoder

LSH-based text-to-spike-pattern encoding. Maps text chunks to neuron
groups via locality-sensitive hashing, generates Poisson spike trains
weighted by chunk salience.

```python
from sc_neurocore.identity import TraceEncoder

encoder = TraceEncoder(n_neurons=500, hash_dims=64)
pattern = encoder.encode("concept to encode", duration_ms=200)
```

### StateDecoder

Extracts cognitive state for session priming: dominant PCA patterns,
stable attractors via correlation clustering, functional connectivity.

```python
from sc_neurocore.identity import StateDecoder

decoder = StateDecoder(substrate)
context = decoder.generate_priming_context()
attractors = decoder.extract_attractor_states(threshold=0.8)
```

### Checkpoint (Lazarus Protocol)

Save/restore/merge complete network state: voltages, CSR weights,
STDP traces, spike history, metadata. Stored as `.npz`.

```python
from sc_neurocore.identity import Checkpoint

Checkpoint.save(substrate, "session_01.npz")
restored = Checkpoint.load("session_01.npz")
merged = Checkpoint.merge(["session_01.npz", "session_02.npz"])
```

### DirectorController (L16)

Self-monitoring and self-regulation. Measures rate, CV, Fano factor,
permutation entropy. Diagnoses problems (rate drift, chaos, silence,
connectivity collapse) and applies corrective actions.

```python
from sc_neurocore.identity import DirectorController

director = DirectorController(substrate)
problems = director.diagnose()
director.correct()
print(director.report())
```
