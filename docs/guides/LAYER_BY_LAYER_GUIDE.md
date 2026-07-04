# Layer-by-Layer Guide: Building Your Architecture

**Version**: 3.16.0
**Target**: Detailed Component Reference

---

## 1. Neuron Types

The `sc_neurocore.neurons` module provides several neuron models tailored for different layers of the SCPN.

### Stochastic LIF (`stochastic_lif.py`)
The "standard" model for L2-L4 simulations. It combines leaky integration with stochastic noise.
*   **Key Params**: `tau_mem`, `v_threshold`, `noise_std`, `v_rest`.
*   **Use Case**: General purpose biological modeling.

### Izhikevich (`sc_izhikevich.py`)
A computationally efficient model that can reproduce a wide variety of firing patterns (regular spiking, bursting, chattering).
*   **Key Params**: `a`, `b`, `c`, `d`.
*   **Use Case**: When specific firing temporal dynamics are required.

### Homeostatic LIF (`homeostatic_lif.py`)
Features a dynamic threshold that increases after spiking and decays over time (refractory period).
*   **Use Case**: Preventing "runaway" activity in deep recurrent networks.

### Fixed-Point LIF (`fixed_point_lif.py`)
Optimized for hardware deployment (PYNQ-Z2), using integer arithmetic instead of floats.

---

## 2. Layer Architectures

Layers in `sc_neurocore` are containers for ensembles of neurons, providing efficient vectorized processing.

### SC Dense Layer (`sc_dense_layer.py`)
A fully connected layer where every input is connected to every neuron.
*   **Weights**: $W \in \mathbb{R}^{N \times M}$
*   **Connectivity**: All-to-all.

### SC Conv Layer (`sc_conv_layer.py`)
Implements spiking convolution, ideal for processing spatial data (L4 Morphology).
*   **Mechanism**: Sliding kernels apply spatial filters to the input spike stream.

### Memristive Layer (`memristive.py`)
Simulates non-volatile memory behavior in synapses (L9 Identity).
*   **Feature**: Weights persist even if the simulation is reset, modeling long-term memory.

### Fusion Layer (`fusion.py`)
A specialised layer designed to integrate same-width modality vectors with
normalised stochastic-computing weights. Positive raw weights are normalised to
sum to one; zero or negative total weights fall back to equal weights across the
weighted modalities, matching the Rust `FusionLayer` fallback. Inputs must be
one-dimensional arrays with their declared feature length, and unweighted
modalities are ignored rather than mixed into the output.
*   **Use Case**: Implementing the "Consilium" (L15) optimisation logic.
*   **Verification**: `tests/test_layers/test_fusion.py` covers normalisation,
    equal-weight fallback, validation failures, output dtype/shape, and Rust
    weighted-sum parity semantics. Local pytest-cov reports 100% line coverage
    for `src/sc_neurocore/layers/fusion.py`. The opt-in non-isolated perf smoke
    `SC_NEUROCORE_PERF=1 ... test_fusion_perf_small` passed on 2026-06-27
    (`1 passed in 1.92s`); no formal isolated benchmark artefact exists for this
    layer yet.

---

## 3. Synaptic Components

Located in `sc_neurocore.synapses`, these define how signals travel between layers.

*   **Static Synapse**: Fixed weights.
*   **STDP Synapse**: Weights evolve based on spike timing.
*   **Quantum Synapse**: Probabilistic transmission based on L1 states.

---

## 4. Connectivity Patterns

### Feed-Forward
Standard L1 -> L2 -> L3 flow. Used for feature extraction.

### Recurrent
Connections within a layer (`recurrent.py`). Necessary for temporal memory and rhythm generation (L8).

### Lateral Inhibition
Neurons within a layer inhibit their neighbors. This "Winner-Take-All" dynamic is used for pattern recognition and focus.

---

## 5. Summary Table

| Component | Module | SCPN Mapping | Complexity |
| :--- | :--- | :--- | :--- |
| **LIF Neuron** | `neurons.stochastic_lif` | L2 (Neural) | Low |
| **Izhikevich** | `neurons.sc_izhikevich` | L2 (Dynamics) | Medium |
| **Dense Layer** | `layers.sc_dense_layer` | L3 (Cognition) | Medium |
| **Fusion Layer** | `layers.fusion` | L15 (Consilium) | High |
| **STDP** | `learning.stdp` | L3 (Learning) | Medium |
| **Director** | `meta.director` | L16 (Oversight) | Extreme |
