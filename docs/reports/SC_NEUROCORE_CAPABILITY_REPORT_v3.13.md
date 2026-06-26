# SC-NeuroCore v3.13.3 Capability Report

**Stochastic Computing Framework for Neuromorphic Hardware Design**

**Version:** 3.13.3
**Report Date:** March 18, 2026
**Author:** Miroslav Šotek (Anulum Research)

---

## Executive Summary

SC-NeuroCore is a Python+Rust framework for designing, simulating, and
deploying spiking neural networks on FPGA hardware via stochastic
computing. It provides bit-true simulation matching synthesisable Verilog
RTL cycle-exactly, surrogate gradient training in PyTorch, and an IR
compiler emitting SystemVerilog.

### Key Metrics

| Metric | Value | Basis |
|--------|-------|-------|
| Neuron models | 122 Python, 111 Rust | Counted |
| Test suite | 3 376+ Python + 373 Rust, 100% coverage | CI-enforced |
| MNIST accuracy | 99.49% (conv SNN) | `benchmarks/results/mnist_conv_accuracy_reproducibility.json` |
| Brunel 1K speedup vs Brian2 | 4.0x | Benchmarked (Numba JIT) |
| Bitstream packing | 41.3 Gbit/s (AVX-512) | Criterion benchmark |
| Synthesis (sc_neurocore_top) | 3 673 LUTs (Xilinx 7-series) | Yosys report |
| Formal verification | Archived v3.13 inventory; current public inventory is 18 proof jobs / 130 formal statements | SymbiYosys |
| CI workflows | 13, all SHA-pinned | GitHub Actions |
| Python versions | 3.10–3.14 | CI matrix |
| Platforms | Linux, macOS, Windows | Wheel builds |

---

## 1. Core Architecture

### 1.1 Stochastic Computing Encoding

Values are represented as Bernoulli bitstreams where the proportion
of 1-bits encodes the value:

```
Value 0.7 → Bitstream: [1,1,0,1,1,1,0,1,1,1,...] (70% ones)
```

Arithmetic maps to logic gates:

| Operation | SC Implementation | Gate Count |
|-----------|-------------------|------------|
| Multiplication | AND gate | 1 |
| Scaled addition | MUX | 3 |
| Integration | Popcount | O(log N) |

This gate reduction is the foundational SC result (Alaghi & Hayes 2013).
The trade-off: precision scales as O(1/√L) where L is bitstream length.
At L=1024, effective precision is ~5 bits.

### 1.2 LFSR-Based Decorrelation

All bitstreams use a 16-bit maximal-length LFSR (polynomial
x^16+x^14+x^13+x^11+1, period 65 535) with deterministic seed
assignment. This ensures reproducible simulation and bit-exact
correspondence with the Verilog RTL implementation.

### 1.3 Fixed-Point Arithmetic

Q8.8 signed two's complement with explicit bit-width masking.
Overflow semantics match the Verilog implementation exactly.

---

## 2. Validated Capabilities

### 2.1 Neuron Model Library (122 models)

117 models in `sc_neurocore.neurons.models` plus 5 core SC neurons
at the package level. Covers:

- **IF variants** (15): LIF, AdEx, ExpIF, Lapicque, GLIF, parametric, homeostatic, etc.
- **Biophysical** (12): Hodgkin-Huxley, TraubMiles, ConnorStevens, WangBuzsaki, etc.
- **Multi-compartment** (5): PinskyRinzel, HayL5, Rall cable, etc.
- **Map-based** (6): Rulkov, Chialvo, Izhikevich, etc.
- **Neural mass** (5): Wilson-Cowan, Jansen-Rit, WongWang, etc.
- **Hardware emulators** (9): Loihi, Loihi2, TrueNorth, BrainScaleS, SpiNNaker, SpiNNaker2, Akida, NeuroGrid, DPI
- **AI-optimized** (9): ArcaneNeuron, AttentionGated, PredictiveCoding, MetaPlastic, etc.
- **Other** (61): bursting, oscillatory, stochastic, population, rate models

All models have Python dataclass implementations with a `step(current) → spike` interface.

### 2.2 Rust SIMD Engine (111 models)

PyO3-bound Rust crate with runtime SIMD detection:

| SIMD Tier | Throughput (bitstream packing) |
|-----------|-------------------------------|
| AVX-512 | 41.3 Gbit/s |
| AVX2 | ~20 Gbit/s |
| NEON (ARM) | ~12 Gbit/s |

`NetworkRunner` provides a fused simulation loop with CSR-sparse
projections and Rayon-parallel population stepping, scaling to 100K+
neurons.

### 2.3 Verilog RTL (17 modules)

Synthesisable Verilog-2005 modules:

| Module | Description |
|--------|-------------|
| `sc_lif_neuron.v` | Q8.8 LIF with configurable threshold and refractory period |
| `sc_dense_matrix_layer.v` | Per-neuron weight matrix |
| `sc_neurocore_top.v` | AXI-Lite wrapper for SoC integration |
| + 14 others | Encoder, synapse, dotproduct, firing rate, etc. |

**Synthesis results** (Yosys, Xilinx 7-series):

| Configuration | LUTs |
|---------------|------|
| sc_neurocore_top | 3 673 |
| MNIST 16→10 (estimated) | ~56 000 |

### 2.4 Formal Verification

Archived v3.13 formal inventory has been superseded. Current public
documentation should cite 18 SymbiYosys proof jobs and 130 formal statements
(100 assert, 7 assume, 23 cover) across the HDL formal tree. The historical
scope covered encoder, neuron, synapse, dense layer, dotproduct, firing rate,
and AXI-Lite config proofs.

### 2.5 Surrogate Gradient Training

PyTorch training module with LIF, adaptive LIF (Bellec 2020), and
recurrent LIF cells. Surrogate gradient backward passes with
learnable membrane and threshold parameters.

| Benchmark | Architecture | Accuracy |
|-----------|-------------|----------|
| MNIST (FC-SNN) | 784→128→128→10, 10 epochs | 95.5% |
| MNIST (FC-SNN + learnable τ) | Same + Fang 2021 | 97.7% |
| MNIST (Conv-SNN) | Conv→LIF→Pool→Conv→LIF→Pool→FC | 99.49% (`benchmarks/results/mnist_conv_accuracy_reproducibility.json`) |

`to_sc_weights()` exports trained float weights normalised to [0,1]
for SC bitstream deployment.

### 2.6 Spike Train Analysis (125 functions)

Pure NumPy, zero external dependencies. Covers:

- Statistics: CV, Fano factor, burst detection
- Distance metrics: Victor-Purpura, van Rossum, SPIKE-distance
- Synchrony: cross-correlation, STTC, event synchronization
- Information theory: mutual information, transfer entropy
- Causality: Granger causality, PDC
- Dimensionality reduction: PCA, GPFA
- Decoding: population vector, Bayesian
- Pattern detection: SPADE

### 2.7 IR Compiler

Graph-based intermediate representation with structural verification
and SystemVerilog emission targeting Xilinx and Intel FPGAs.

### 2.8 Network Simulation Engine

Population-Projection-Network architecture with three backends:

| Backend | Scale | Speed |
|---------|-------|-------|
| Python (NumPy) | ≤5K neurons | Baseline |
| Rust (NetworkRunner) | ≤100K neurons | ~50x Python |
| MPI (mpi4py) | Billion-neuron | Distributed |

Six topology generators: random, small-world, scale-free, ring,
grid, all-to-all.

Model zoo: 10 pre-built configurations + 3 pre-trained weight sets
(MNIST, SHD speech, DVS gesture).

---

## 3. Benchmarks

### 3.1 Brunel Balanced Network (vs Brian2)

| Network size | SC-NeuroCore (Numba) | Brian2 (C++ codegen) | Ratio |
|-------------|---------------------|---------------------|-------|
| 1 000 neurons | 0.35 s | 1.38 s | 4.0x faster |
| 10 000 neurons | 5.9 s | 4.4 s | 1.35x slower |

Firing rates match within 1% (100 Hz). SC-NeuroCore targets
FPGA-scale networks (≤5K neurons) where bit-exact RTL co-simulation
matters; Brian2 scales better for large sparse networks.

### 3.2 Fault Tolerance

SC bitstreams degrade gracefully under random bit errors:

| Error rate | Accuracy loss (at p=0.5) |
|------------|-------------------------|
| 1% | <1% |
| 5% | <1% |
| 10% | ~1% |

At balanced probability (p=0.5), errors are symmetric and partially
self-canceling. At extreme probabilities (p near 0 or 1), degradation
is more significant. This is a property of stochastic encoding, not
a SC-NeuroCore-specific achievement.

### 3.3 HDC/VSA Pattern Capacity

10 000-bit hyperdimensional vectors with XOR binding and majority-vote
bundling. Hamming distance query over 100 stored patterns completes
in ~1 ms. Noise tolerance: correct retrieval at 10% bit noise.

---

## 4. Experimental Modules

The following modules exist as working code but are not part of the
core production API. They are research prototypes with limited testing.

| Module | Purpose | Status |
|--------|---------|--------|
| `quantum/hybrid.py` | Quantum-classical hybrid layer (Qiskit/PennyLane) | Functional, requires quantum backend |
| `hdc/base.py` | Hyperdimensional computing encoder and memory | Functional, tested |
| `transformers/block.py` | Stochastic transformer block | Prototype |
| `graphs/gnn.py` | Stochastic graph neural network layer | Prototype |
| `solvers/ising.py` | Ising machine for combinatorial optimization | Prototype |
| `world_model/predictive_model.py` | Model-based planning | Prototype |
| `interfaces/dvs_input.py` | DVS event camera input | Functional |
| `interfaces/bci.py` | BCI signal encoder | Prototype |
| `optics/photonic_layer.py` | Photonic interference simulation | Simulation only |
| `bio/dna_storage.py` | DNA encoding scheme | Encoder only, no wet lab |

### Energy estimates

The gate-level energy model estimates 5.10 fJ per SC bit-operation
based on published CMOS gate energies. This is a model estimate,
not a measured result on fabricated silicon. Actual energy depends
on technology node, clock frequency, and routing.

---

## 5. Identity Substrate

`sc_neurocore.identity` provides a persistent spiking network for
identity continuity research:

- 3-population architecture (HH cortical, WB inhibitory, HR memory)
- STDP-connected via small-world topology
- Text-to-spike encoding (LSH), state decoding (PCA + attractor extraction)
- Checkpoint save/restore (Lazarus protocol)
- L16 Director controller for cybernetic self-regulation
- 9 AI-optimized neuron models including ArcaneNeuron

This is a research module, not a production inference tool.

---

## 6. Quality Assurance

| Gate | Tool | Threshold |
|------|------|-----------|
| Python tests | pytest | 3 376+ tests, 100% line coverage |
| Rust tests | cargo test | 373 tests |
| Formatting | ruff format 0.15.6 | 529 files |
| Linting | ruff 0.15.6 | Zero violations |
| Security | bandit | Zero findings |
| SPDX headers | CI guard | All .py, .rs, .v files |
| Formal | SymbiYosys | Current public inventory: 18 proof jobs / 130 formal statements |
| Supply chain | CodeQL + OpenSSF Scorecard | Active |
| CI workflows | 13, all SHA-pinned | Every push |

---

## 7. Limitations

- **Precision**: SC bitstream precision scales as O(1/√L). At L=1024,
  effective precision is ~5 bits. Not suitable for applications
  requiring float32 accuracy.
- **Scale**: Bit-true co-simulation is practical for ≤5K neurons.
  Larger networks should use the Rust NetworkRunner without RTL
  correspondence.
- **FPGA deployment**: No physical FPGA run has been completed. The
  Yosys synthesis reports are from open-source tools; Vivado
  place-and-route results may differ.
- **Training**: The surrogate gradient module matches snnTorch on
  standard benchmarks but does not exceed it. The value is the
  hardware export path, not training performance.
- **Rust parity**: 12 of 122 neuron models have known Rust/Python
  parity divergences (tracked as xfail in CI).
- **Brian2 scaling**: At >5K neurons, Brian2's compiled C++ codegen
  is faster than SC-NeuroCore's JIT backend.

---

## 8. Installation

```bash
pip install sc-neurocore          # Core (NumPy + SciPy)
pip install sc-neurocore[accel]   # + Numba JIT
pip install sc-neurocore[research] # + PyTorch, matplotlib
pip install sc-neurocore[full]    # Everything
```

Python 3.10–3.14. Linux, macOS, Windows.

---

## 9. Citation

```bibtex
@software{scneurocore2026,
  title={SC-NeuroCore: A Deterministic Stochastic Computing Framework
         for Neuromorphic Hardware Design},
  author={Šotek, Miroslav},
  version={3.13.3},
  year={2026},
  doi={10.5281/zenodo.18906614},
  url={https://github.com/anulum/sc-neurocore}
}
```

---

*SC-NeuroCore is developed by [Anulum Research](https://www.anulum.li).
AGPL-3.0-or-later with commercial license option.
Contact: neurocore@anulum.li*
