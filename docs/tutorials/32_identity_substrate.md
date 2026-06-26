<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 32: Identity Substrate — Persistent Spiking Networks

The identity substrate is a persistent spiking neural network that maintains
state across sessions. It encodes experiences as spike patterns, stores them
in STDP-modified synaptic weights, and can be saved, restored, and merged
via the Lazarus checkpoint protocol. A cybernetic Director controller monitors
and regulates network dynamics.

## Architecture

The substrate uses three biologically distinct neuron populations:

| Population | Model | Count | Role |
|------------|-------|------:|------|
| Cortical | Hodgkin-Huxley | 500 | Fast processing, pattern recognition |
| Inhibitory | Wang-Buzsaki | 200 | Lateral inhibition, oscillation control |
| Memory | Hindmarsh-Rose | 100 | Bursting dynamics, long-term trace storage |

Populations are connected with small-world topology and STDP-enabled projections.

## 1. Create and Run

```python
from sc_neurocore.identity.substrate import IdentitySubstrate

# Create substrate with default parameters
substrate = IdentitySubstrate(
    n_cortical=500,
    n_inhibitory=200,
    n_memory=100,
    seed=42,
)

# Run for 1 second of simulated time
substrate.run(duration=1.0, dt=0.001)

# Check health
health = substrate.health_check()
print(f"Mean rate: {health['mean_rate']:.1f} Hz")
print(f"CV: {health['cv']:.2f}")
print(f"Healthy: {health['is_healthy']}")
```

## 2. Inject Experiences

Text is converted to spike patterns via locality-sensitive hashing (LSH):

```python
from sc_neurocore.identity.encoder import TraceEncoder

encoder = TraceEncoder(n_neurons=500, hash_dims=64, seed=42)

# Encode a reasoning trace as a spike pattern
pattern = encoder.encode("The Euler method converges as O(dt)")
print(f"Pattern shape: {pattern.shape}")  # (100, 500) — 100ms x 500 neurons

# Inject into substrate — STDP modifies weights based on the pattern
substrate.inject_experience("The Euler method converges as O(dt)")

# Multiple experiences build up weight structure
substrate.inject_experience("NIR bridge maps 18 primitives to SC nodes")
substrate.inject_experience("Formal verification covers 18 proof jobs")
```

## 3. Extract State

The decoder extracts high-level structure from the substrate's spiking activity:

```python
from sc_neurocore.identity.decoder import StateDecoder

decoder = StateDecoder(substrate)

# PCA on recent spike trains — dominant activity patterns
patterns = decoder.extract_dominant_patterns(n_components=10)
print(f"Dominant patterns: {patterns.shape}")

# Attractor states — stable activity configurations
attractors = decoder.extract_attractor_states(threshold=0.8)
print(f"Found {len(attractors)} attractor states")

# Connectivity signature — fingerprint of the weight matrix
signature = decoder.extract_connectivity_signature()
print(f"Connectivity signature: {signature.shape}")

# Generate a text summary of the current state
context = decoder.generate_priming_context()
print(context)
```

## 4. Checkpoint: Save, Restore, Merge

The Lazarus protocol preserves complete network state in `.npz` files:

```python
from sc_neurocore.identity.checkpoint import Checkpoint

# Save current state
Checkpoint.save(substrate, "session_001.npz")

# Continue working, inject more experiences...
substrate.inject_experience("CubaLIF roundtrip preserves all 7 parameters")
substrate.run(duration=2.0, dt=0.001)

# Save again
Checkpoint.save(substrate, "session_002.npz")

# Restore from any checkpoint
restored = Checkpoint.load("session_001.npz")

# Merge multiple sessions — combines weight matrices
merged = Checkpoint.merge(["session_001.npz", "session_002.npz"])
```

## 5. Director Controller

The L16 cybernetic controller monitors substrate health and applies corrections:

```python
from sc_neurocore.identity.director import DirectorController

director = DirectorController(substrate)

# Monitor: get network metrics
metrics = director.monitor()
print(f"Metrics: {metrics}")

# Diagnose: list any problems
problems = director.diagnose()
if problems:
    for p in problems:
        print(f"Issue: {p}")
else:
    print("No issues detected")

# Correct: apply interventions (modifies substrate in-place)
director.correct()
```

## 6. Full Lifecycle Example

```python
from sc_neurocore.identity.substrate import IdentitySubstrate
from sc_neurocore.identity.encoder import TraceEncoder
from sc_neurocore.identity.decoder import StateDecoder
from sc_neurocore.identity.checkpoint import Checkpoint

# Session 1: create and learn
substrate = IdentitySubstrate(seed=42)
substrate.run(duration=0.5, dt=0.001)  # warm up

# Inject experiences from a work session
experiences = [
    "sc-neurocore has 158 lazy-loaded Python model classes",
    "The Rust engine achieves 113 Gbit/s on AVX-512",
    "NIR bridge verified with Norse, snnTorch, SpikingJelly",
]
for exp in experiences:
    substrate.inject_experience(exp)
    substrate.run(duration=0.2, dt=0.001)

# Save
Checkpoint.save(substrate, "identity_v1.npz")

# Session 2: restore and continue
substrate = Checkpoint.load("identity_v1.npz")
decoder = StateDecoder(substrate)
print("Restored context:", decoder.generate_priming_context())

# The weight structure from Session 1 persists
# New experiences build on the existing patterns
substrate.inject_experience("Equation compiler maps ODEs to Q8.8 Verilog")
Checkpoint.save(substrate, "identity_v2.npz")
```

## How It Works

1. **Encoding**: LSH maps text tokens to neuron indices. Each token activates a
   sparse subset of cortical neurons. Repeated tokens activate the same neurons.

2. **Storage**: STDP modifies excitatory-excitatory weights based on spike timing.
   Co-active neurons strengthen their connections. The weight matrix IS the memory.

3. **Retrieval**: Inject a partial pattern (a few tokens). The network's attractor
   dynamics complete the pattern — neurons that were co-active during encoding
   fire together again.

4. **Persistence**: The `.npz` checkpoint stores all neuron states (voltages,
   recovery variables) and all synaptic weights. Restoring a checkpoint resumes
   the exact dynamical state.

## Further Reading

- [API: Identity Substrate](../api/identity.md) — auto-generated API docs
- [Tutorial 24: Biological Circuits](24_biological_circuits.md) — the neuron models used
- [Tutorial 08: Online Learning](08_online_learning_stdp.md) — STDP mechanics
