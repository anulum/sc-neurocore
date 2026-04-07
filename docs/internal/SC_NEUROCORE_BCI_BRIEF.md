# SC-NeuroCore — Technical Brief for BCI Applications

**Prepared by:** Anulum Research (Swiss/Liechtenstein/Slovak Private Research Institute)
**Contact:** protoscience@anulum.li | www.anulum.li
**Date:** 2026-04-07 (revised)
**Classification:** Confidential — for evaluation purposes

---

## Executive Summary

SC-NeuroCore is a stochastic computing SNN framework that compiles
trained spiking neural networks to synthesisable Verilog with bit-true
Python-to-hardware co-simulation. This brief covers only the modules
directly relevant to brain-computer interface (BCI) implant design.

**Competition-free position:** No other SNN framework generates
synthesisable RTL from trained models. snnTorch, Norse, BindsNET are
GPU-only; Lava is locked to Intel Loihi; Brian2 generates C++ but no
RTL. SC-NeuroCore is the only open path from a trained SNN to FPGA/ASIC
gates with formal verification.

---

## 1. Neural Data Compression

**Problem:** 1024+ electrode arrays at 30 kHz produce ~60 MB/s raw data.
Bluetooth uplink bandwidth is ~2 Mbps. On-chip compression is mandatory.

**SC-NeuroCore solution:**

### WaveformCodec — three compression modes

End-to-end pipeline: spike detection → template matching → spatial
decorrelation → wavelet denoising → entropy coding. Three modes
targeting different BCI requirements:

| Mode | What is preserved | 1024-ch ratio | 384-ch ratio |
|------|-------------------|---------------|-------------|
| `spike` | Spike timing (lossless) | **4,569x** | 4,011x |
| `waveform` | + waveform templates | **1,681x** | 1,320x |
| `full` | + background LFP (lossy) | **137x** | 110x |

Measured on spatially correlated data (exponential covariance, 40 µm
length constant, 0.62 adjacent-channel correlation, 30 kHz, 10-bit
ADC, biphasic spike waveforms with spatial spread). Seed 42,
fully reproducible.

**Comparison with Neuralink N1:** N1 on-chip spike detection achieves
200x. In the equivalent spike-only mode, SC-NeuroCore achieves
**4,569x — 23x better**. Additionally, waveform and full modes
preserve information that N1 discards.

### Electrode scaling and Bluetooth fit

| Channels | Raw data | spike mode | BT 2 Mbps fit | waveform mode | BT fit |
|----------|---------|-----------|---------------|--------------|--------|
| 1,024 | 491 Mbps | 0.11 Mbps | YES (95%) | 0.28 Mbps | YES |
| 4,096 | 1,966 Mbps | 0.43 Mbps | YES (79%) | 1.13 Mbps | YES |
| 16,384 | 7,864 Mbps | 1.72 Mbps | YES (14%) | 4.54 Mbps | NO |

**Spike mode fits Bluetooth at any foreseeable electrode count**
including the Neuralink N3 target of 16,384 electrodes.

### Spike raster codecs (6 algorithms)

For pre-sorted binary rasters (post spike detection):

| Codec | Compression ratio | Latency | Best for |
|-------|------------------|---------|----------|
| ISI + Huffman | 50–200x | Low | Sparse activity |
| Predictive (4 learnable predictors) | 100–750x | Medium | Patterned activity |
| Delta | 30–100x | Minimal | Streaming |
| Streaming | 50–150x | Real-time | Continuous telemetry |
| AER (Address-Event) | 20–80x | Minimal | Event-driven hardware |

Unified API: `get_codec(name)`, `recommend_codec()` auto-selects based
on spike density and latency requirements.

Learnable world-model predictor achieves 99.6% prediction accuracy.
Bit-true LFSR matches Verilog RTL (co-simulation verified).

---

## 2. Equation-to-Verilog Compiler

**Problem:** Custom on-chip signal processing blocks require hand-written
RTL. Iteration cycle: weeks.

**SC-NeuroCore solution:**

Input: arbitrary ODE system as a string:
```python
from sc_neurocore.compiler import ode_to_verilog

verilog = ode_to_verilog(
    "dv/dt = (-v + R*I) / tau",
    params={"R": 1.0, "tau": 10.0},
    precision="Q8.8",  # configurable 2-16 bit
    target="asic"       # or "ice40", "xilinx"
)
```

Output: synthesisable Verilog module with:
- Q8.8 fixed-point arithmetic (configurable precision via QAT, int8 supported)
- Deterministic LFSR-based stochastic encoding
- Cycle-exact Python-to-RTL co-simulation
- Automatic pipeline register insertion for timing closure

**Supported targets:** iCE40 (Yosys+nextpnr, fully open), Xilinx (Vivado
project files), generic ASIC (technology-independent RTL).

---

## 3. Event-Driven FPGA Architecture

**Problem:** Clock-driven designs waste power toggling registers every
cycle, even when no spikes occur. At 0.1% average spike rate, 99.9% of
toggles are wasted.

**SC-NeuroCore solution:**

Three event-driven RTL primitives:
- **AER Encoder:** Address-Event Representation — only active neurons
  generate bus transactions
- **Event Neuron:** Processes only on input events, sleeps otherwise
- **Spike Router:** Configurable point-to-point and multicast routing

**Measured results:**
- 15-39x fewer register toggles than clock-driven at 0.01-10% activity
- Power reduction scales with spike sparsity (typical BCI: <1% activity)
- Verified via toggle-count analysis in simulation

---

## 4. Population Decoders

**Problem:** Real-time decoding of motor intent from neural populations
for cursor/prosthetic control.

**SC-NeuroCore solution:** 4 publication-exact foundation-model decoders:

| Decoder | Architecture | Causal | Latency | Paper |
|---------|-------------|--------|---------|-------|
| POYODecoder | Cross-attention (PerceiverIO) | No | Batch | Azabou et al. 2023, NeurIPS |
| **POSSMDecoder** | Diagonal SSM (S4D) | **Yes** | **O(1)/step** | Ryoo et al. 2025, ICLR |
| NDT3Decoder | Causal masked attention | Yes | O(T) | Ye & Pandarinath 2025 |
| CEBRAEncoder | Contrastive MLP | — | O(1) | Schneider et al. 2023, Nature |

**POSSMDecoder is most relevant for real-time BCI:** constant-time per
step, causal (no future information leakage), 9x faster than transformer
attention. State-space model with HiPPO-LegS initialisation captures
multi-scale temporal dynamics.

All decoders have Rust-accelerated kernels (tokenise_spikes,
scaled_dot_product_attention, ssm_step_diagonal, infonce_loss).

---

## 5. Spike Train Analysis Toolkit

132 functions across 24 modules. Relevant subset for BCI:

| Function | Purpose |
|----------|---------|
| `firing_rate`, `instantaneous_rate` | Activity monitoring |
| `burst_detection` | Pathological activity detection |
| `spike_train_pca`, `gpfa` | Dimensionality reduction |
| `mutual_information`, `transfer_entropy` | Channel selection |
| `cross_correlation`, `spike_train_coherence` | Functional connectivity |
| `isolation_distance`, `l_ratio`, `snr` | Spike sorting quality metrics |
| `waveform_width`, `waveform_amplitude` | Unit classification |

All functions have Rust acceleration via PyO3.

---

## 6. Formal Verification

**Problem:** Implant hardware must be provably correct. Simulation cannot
cover all corner cases.

**SC-NeuroCore solution:**

7 SymbiYosys formal verification modules with 72 proven properties:
- Bitstream encoder correctness (LFSR period, output bounds)
- Neuron state boundedness (no overflow, no NaN equivalent)
- AER protocol compliance (valid address, handshake timing)
- Pipeline data integrity (no data loss, no corruption)

Properties are proven for ALL possible inputs, not just test vectors.
Uses bounded model checking and k-induction.

---

## 7. Performance

### Rust SIMD Engine
| Metric | Value | Hardware |
|--------|-------|----------|
| Single LIF step | 3.8 ns | i5-11600K |
| 100K neuron network | 1.15 s (300 ms sim) | i5-11600K |
| Synaptic throughput | 27.7 B events/s | i5-11600K |
| vs Brian2 speedup | 39-202x | Same hardware |
| SIMD popcount | 190 Gbit/s | AVX-512 |

### Waveform Compression (1024-ch, 30 kHz, 10-bit, correlated)
| Metric | Value |
|--------|-------|
| Spike mode | 4,569x (lossless spike timing) |
| Waveform mode | 1,681x (+ templates) |
| Full mode | 137x (+ background LFP) |
| Bluetooth fit (spike) | Yes — up to 16,384 electrodes |
| Bluetooth fit (waveform) | Yes — up to ~7,000 electrodes |

### Test Coverage
| Suite | Count |
|-------|-------|
| Python tests | 7,425 |
| Rust tests | 1,549 |
| Formal properties | 72 |

---

## 8. Computational Bottlenecks and Roadmap

### Current (Python)

| Step | Time (1024-ch, 1s) | % |
|------|-------------------|---|
| Spike detection | 9,266 ms | 72% |
| Noise estimation | 1,666 ms | 13% |
| Background compression | 1,222 ms | 10% |
| Other | 637 ms | 5% |
| **Total** | **12,791 ms** | |

Real-time factor: 0.08x (not real-time in Python).

### Planned (Rust SIMD)

Spike detection inner loop is a threshold comparison with refractory
check — perfectly vectorisable with AVX2/AVX-512. Based on existing
Rust SIMD benchmarks in the codebase (3.8 ns/sample LIF, 190 Gbit/s
popcount):

| Backend | Estimated total | Real-time factor | Max channels |
|---------|----------------|-----------------|-------------|
| Python | 12,791 ms | 0.08x | ~80 |
| Rust | ~700 ms | ~1.4x | ~4,000 |
| Rust + SIMD | ~65 ms | ~15x | ~16,000 |

**Neuralink N3 target (16,384 electrodes) is achievable with Rust + SIMD.**

---

## 9. What Is NOT in This Brief

The following SC-NeuroCore modules exist but are not relevant to BCI
implant design and are omitted from this brief:

- 178 neuron model library (simulation, not implant)
- SCPN theoretical framework (consciousness research)
- Quantum hybrid computing (Qiskit/PennyLane)
- Hyperdimensional computing
- Identity substrate / world model
- Visual SNN Design Studio
- ANN-to-SNN conversion pipeline
- Training infrastructure (surrogate gradient, STDP, etc.)

These are available for evaluation on request.

---

## 10. Availability

- **Repository:** Transitioning to private access. Full access granted
  for evaluation under NDA.
- **License:** Dual — AGPL-3.0 (academic/research), commercial license
  available.
- **Contact:** protoscience@anulum.li
- **Web:** www.anulum.li
