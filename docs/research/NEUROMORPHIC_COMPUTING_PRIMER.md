# Neuromorphic Computing Primer: The Philosophy of Spikes

**Version:** 3.15.30
**Target**: Conceptual Introduction

---

## 1. Why Neuromorphic?

Traditional Artificial Intelligence (like Transformers or CNNs) operates on continuous numbers and high-precision floating-point arithmetic. Biological brains, however, operate using **discrete pulses of electricity** called **spikes** (action potentials).

**sc-neurocore** is a neuromorphic framework designed to bridge this gap. By modeling neurons as discrete spike-generators, we achieve several advantages:

1.  **Temporal Precision**: Information is encoded in the *timing* of spikes, not just their magnitude.
2.  **Power Efficiency**: In hardware (like the PYNQ-Z2), neurons only "consume" energy when they spike. Most of the time, the system is silent.
3.  **Biological Realism**: It allows us to simulate the lower layers of the SCPN (L1-L4) with high fidelity, capturing the "jitter" and "resonance" that continuous models miss.

---

## 2. Key Concepts

### Neurons: The Leaky Integrator
Imagine a bucket with a small hole in the bottom. Water (input current) flows in. If the water reaches the top, the bucket tips over (spikes) and empties (resets). The hole represents the **leak**, ensuring that if input stops, the "voltage" eventually returns to zero.

### Synapses: The Weighted Bridge
Information travels between neurons via synapses. A synapse has a **weight**. If a pre-synaptic neuron spikes, the post-synaptic neuron receives a "kick" proportional to that weight. In `sc-neurocore`, weights can be static or adaptive (learning).

### Learning: STDP
**Spike-Timing-Dependent Plasticity (STDP)** is a biological learning rule.
*   If Neuron A spikes just *before* Neuron B, the connection between them is strengthened (Causality).
*   If Neuron A spikes *after* Neuron B, the connection is weakened.
*   This allows the network to learn temporal patterns without a global "error" signal (backpropagation).

---

## 3. Stochasticity: Embracing the Noise

A unique feature of `sc-neurocore` is its focus on **Stochastic Computing**. Every neuron has a `noise_std` parameter. Why?

In the SCPN framework, Layer 1 is a quantum substrate. Quantum events are probabilistic. By injecting noise into our neuromorphic layers, we maintain the "statistical pressure" required for the higher layers (like the Consilium) to perform global optimization. Noise isn't a bug; it's a feature that prevents the system from getting stuck in local minima.

---

## 4. Relationship to SCPN

`sc-neurocore` serves as the high-performance hardware backend for the SCPN.
*   **L1-L2**: Directly mapped to stochastic neurons and spintronic synapses.
*   **L3-L4**: Implemented as structural constraints on the neuromorphic mesh.
*   **L15-L16**: The Director AI monitors the "total spike entropy" of the neurocore to judge system health.

---

## 5. Summary

Neuromorphic computing isn't just a faster way to do AI—it's a fundamentally different way of processing information that mirrors the structure of the universe as modeled by the SCPN.
