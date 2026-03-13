# Rust Engine — Python Parity Roadmap

Feature parity status between the Rust engine (`engine/`) and the
Python reference implementation (`src/sc_neurocore/`).

## Legend

- **Done** — Rust implementation exists with tests
- **Partial** — Rust code exists but missing edge cases or variants
- **Missing** — No Rust implementation yet

## Core Primitives

| Feature | Python module | Rust module | Status |
|---------|--------------|-------------|--------|
| Bernoulli bitstream | `utils.bitstreams` | `bitstream.rs` | Done |
| Sobol bitstream | `utils.bitstreams` | — | Missing |
| Bitstream pack/unpack | `accel.vector_ops` | `bitstream.rs` | Done |
| Popcount (SWAR) | `accel.vector_ops` | `bitstream.rs` | Done |
| Bitwise AND | `accel.vector_ops` | `bitstream.rs` | Done |
| SIMD popcount (AVX2) | — | `simd/avx2.rs` | Done |
| SIMD popcount (AVX-512) | — | `simd/avx512.rs` | Done |
| SIMD popcount (NEON) | — | `simd/neon.rs` | Done |
| SIMD popcount (SVE) | — | `simd/sve.rs` | Done |
| SIMD popcount (RISC-V V) | — | `simd/rvv.rs` | Done |
| BitstreamEncoder | `utils.bitstreams` | `encoder.rs` | Done |
| BitstreamAverager | `utils.bitstreams` | `neuron.rs` | Done |
| RNG wrapper | `utils.rng` | uses `rand` crate | Done |

## Neurons

| Feature | Python class | Rust struct | Status |
|---------|-------------|-------------|--------|
| LIF neuron | `StochasticLIFNeuron` | `neuron.rs::FixedPointLif` | Done |
| Fixed-point LIF | `FixedPointLIFNeuron` | `neuron.rs::FixedPointLif` | Done |
| LFSR-16 | `FixedPointLFSR` | `encoder.rs::Lfsr16` | Done |
| FP encoder | `FixedPointBitstreamEncoder` | `encoder.rs::BitstreamEncoder` | Done |
| Izhikevich | `SCIzhikevichNeuron` | `neuron.rs::Izhikevich` | Done |
| Homeostatic LIF | `HomeostaticLIFNeuron` | — | Missing |
| Dendritic neuron | `DendriticNeuron` | — | Missing |

## Synapses

| Feature | Python class | Rust struct | Status |
|---------|-------------|-------------|--------|
| Static synapse | `BitstreamSynapse` | `synapses/mod.rs` | Partial |
| STDP synapse | `StochasticSTDPSynapse` | `synapses/mod.rs::StdpSynapse` | Done |
| R-STDP synapse | `RewardModulatedSTDPSynapse` | — | Missing |
| Dot product | `BitstreamDotProduct` | — | Missing |

## Layers

| Feature | Python class | Rust struct | Status |
|---------|-------------|-------------|--------|
| Dense layer | `SCDenseLayer` | `layer.rs` | Done |
| Vectorized layer | `VectorizedSCLayer` | `layer.rs` (packed) | Done |
| Conv2D layer | `SCConv2DLayer` | — | Missing |
| Recurrent layer | `SCRecurrentLayer` | — | Missing |
| Learning layer | `SCLearningLayer` | — | Missing |
| Fusion layer | `SCFusionLayer` | — | Missing |
| Attention | `StochasticAttentionLayer` | `attention.rs` | Done |
| Memristive | `MemristiveSCLayer` | — | Missing |

## Networks & Analysis

| Feature | Python module | Rust module | Status |
|---------|--------------|-------------|--------|
| Brunel network | `models.zoo` | `brunel.rs` | Done |
| GNN layer | `graphs.gnn` | `graph.rs` | Done |
| Spike recorder | `recorders.spike_recorder` | — | Missing |
| Connectome gen | `utils.connectomes` | — | Missing |
| Fault injection | `utils.fault_injection` | — | Missing |

## Compiler / IR

| Feature | Python module | Rust module | Status |
|---------|--------------|-------------|--------|
| IR builder | `compiler.pipeline` | `ir/builder.rs` | Done |
| IR parser | `compiler.pipeline` | `ir/parser.rs` | Done |
| IR verifier | — | `ir/verify.rs` | Done |
| SystemVerilog emit | — | `ir/emit_sv.rs` | Done |
| IR printer | — | `ir/printer.rs` | Done |
| MLIR emitter | `compiler.mlir_emitter` | — | Missing |

## Surrogate Gradient Training

| Feature | Python module | Rust module | Status |
|---------|--------------|-------------|--------|
| Surrogate grad | `learning.*` (PyTorch) | `grad/surrogate.rs` | Partial |

## Summary

| Category | Done | Partial | Missing | Total |
|----------|------|---------|---------|-------|
| Primitives | 12 | 0 | 1 | 13 |
| Neurons | 5 | 0 | 2 | 7 |
| Synapses | 1 | 1 | 2 | 4 |
| Layers | 3 | 0 | 5 | 8 |
| Networks | 2 | 0 | 3 | 5 |
| Compiler | 5 | 0 | 1 | 6 |
| Training | 0 | 1 | 0 | 1 |
| **Total** | **28** | **2** | **14** | **44** |

**Parity: 68% (30/44 at least partial)**

## Priority Queue

1. `HomeostaticLIFNeuron` — needed for adaptive threshold experiments
2. `SCConv2DLayer` — needed for vision benchmarks
3. `SCRecurrentLayer` — needed for temporal benchmarks
4. `BitstreamSpikeRecorder` — needed for analysis parity
5. `RewardModulatedSTDPSynapse` — needed for reinforcement learning
6. `DendriticNeuron` — needed for compartmental models
