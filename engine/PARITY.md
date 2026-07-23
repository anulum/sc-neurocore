# Rust Engine — Python Parity Roadmap

Feature parity status between the Rust engine (`engine/`) and the
Python reference implementation (`src/sc_neurocore/`).

## Legend

- **Done** — Rust implementation exists with tests

## Core Primitives

| Feature | Python module | Rust module | Status |
|---------|--------------|-------------|--------|
| Bernoulli bitstream | `utils.bitstreams` | `bitstream.rs` | Done |
| Sobol bitstream | `utils.bitstreams` | `sobol.rs` | Done |
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

## Neurons (Core)

| Feature | Python class | Rust struct | Status |
|---------|-------------|-------------|--------|
| LIF neuron | `StochasticLIFNeuron` | `neuron.rs::FixedPointLif` | Done |
| Fixed-point LIF | `FixedPointLIFNeuron` | `neuron.rs::FixedPointLif` | Done |
| LFSR-16 | `FixedPointLFSR` | `encoder.rs::Lfsr16` | Done |
| FP encoder | `FixedPointBitstreamEncoder` | `encoder.rs::BitstreamEncoder` | Done |
| Izhikevich | `SCIzhikevichNeuron` | `neuron.rs::Izhikevich` | Done |
| Homeostatic LIF | `HomeostaticLIFNeuron` | `neuron.rs::HomeostaticLif` | Done |
| Dendritic neuron | `DendriticNeuron` | `neuron.rs::DendriticNeuron` | Done |

## Extended Neuron Models (106 bio + 1 AI in `neurons/`)

| Category | Count | Rust module | Status |
|----------|-------|-------------|--------|
| Trivial IF variants | 18 | `neurons/trivial/` + facade | Done |
| Simple spiking (2D+) | 22 | `neurons/simple_spiking/` + facade | Done |
| Discrete maps | 6 | `neurons/maps.rs` | Done |
| Biophysical (HH-type) | 20 | `neurons/biophysical/` + facade | Done |
| Multi-compartment | 7 | `neurons/multi_compartment/` + facade | Done |
| Stochastic/population | 13 | `neurons/special.rs` | Done |
| Hardware emulators | 9 | `neurons/hardware/` + facade | Done |
| Rate/other | 11 | `neurons/rate/` | Done |
| AI-optimized | 1 | `neurons/special.rs` (Arcane) | Done |

## Synapses

| Feature | Python class | Rust struct | Status |
|---------|-------------|-------------|--------|
| Static synapse | `BitstreamSynapse` | `synapses/mod.rs::StaticSynapse` | Done |
| STDP synapse | `StochasticSTDPSynapse` | `synapses/mod.rs::StdpSynapse` | Done |
| R-STDP synapse | `RewardModulatedSTDPSynapse` | `synapses/mod.rs::RewardStdpSynapse` | Done |

## Layers

| Feature | Python class | Rust struct | Status |
|---------|-------------|-------------|--------|
| Dense layer | `SCDenseLayer` | `layer.rs::DenseLayer` | Done |
| Vectorized layer | `VectorizedSCLayer` | `layer.rs` (packed) | Done |
| Conv2D layer | `SCConv2DLayer` | `conv.rs::Conv2DLayer` | Done |
| Recurrent layer | `SCRecurrentLayer` | `recurrent.rs::RecurrentLayer` | Done |
| Learning layer | `SCLearningLayer` | `fusion.rs::LearningLayer` | Done |
| Fusion layer | `SCFusionLayer` | `fusion.rs::FusionLayer` | Done |
| Attention | `StochasticAttentionLayer` | `attention.rs` | Done |
| Memristive | `MemristiveSCLayer` | `fusion.rs::MemristiveLayer` | Done |

## Networks & Analysis

| Feature | Python module | Rust module | Status |
|---------|--------------|-------------|--------|
| Brunel network | `models.zoo` | `brunel.rs` | Done |
| GNN layer | `graphs.gnn` | `graph.rs` | Done |
| Spike recorder | `recorders.spike_recorder` | `recorder.rs` | Done |
| Connectome gen | `utils.connectomes` | `connectome.rs` | Done |
| Fault injection | `utils.fault_injection` | `fault.rs` | Done |

## Compiler / IR

| Feature | Python module | Rust module | Status |
|---------|--------------|-------------|--------|
| IR builder | `compiler.pipeline` | `ir/builder.rs` | Done |
| IR parser | `compiler.pipeline` | `ir/parser.rs` | Done |
| IR verifier | — | `ir/verify.rs` | Done |
| SystemVerilog emit | — | `ir/emit_sv.rs` | Done |
| IR printer | — | `ir/printer.rs` | Done |
| MLIR emitter | `compiler.mlir_emitter` | `ir/emit_mlir.rs` | Done |

## Surrogate Gradient Training

| Feature | Python module | Rust module | Status |
|---------|--------------|-------------|--------|
| Surrogate grad | `learning.*` (PyTorch) | `grad/surrogate.rs` | Done |

## Summary

| Category | Done | Total |
|----------|------|-------|
| Primitives | 13 | 13 |
| Neurons (core) | 7 | 7 |
| Neurons (extended) | 105 | 105 |
| Synapses | 3 | 3 |
| Layers | 8 | 8 |
| Networks | 5 | 5 |
| Compiler | 6 | 6 |
| Training | 1 | 1 |
| **Total** | **148** | **148** |

**Feature parity: 148/148**
**Rust neuron models callable from Python: 111 of 122 Python total**
**11 Python-only models: 8 AI-optimized (ai_optimized.py) + 3 core name mappings (StochasticLIF, FixedPointLIF, SCIzhikevich)**
**Rust tests: 373 (across 17 test binaries)**
**NetworkRunner: 81 models in fused Rayon-parallel simulation loop**
