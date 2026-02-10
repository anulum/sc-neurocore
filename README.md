# SC-NeuroCore v3.6

[![CI](https://github.com/anulum/sc-neurocore/actions/workflows/v3-engine.yml/badge.svg)](https://github.com/anulum/sc-neurocore/actions)
[![PyPI](https://img.shields.io/pypi/v/sc-neurocore.svg)](https://pypi.org/project/sc-neurocore/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18594898.svg)](https://doi.org/10.5281/zenodo.18594898)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Hardware: Verified](https://img.shields.io/badge/Hardware-Verified_Bit--True-green)](tests/cosim)

**The Industry’s First Verified Rust-Based Neuromorphic Compiler.**

> **Bridging the Gap:** SC-NeuroCore translates high-level Python SNN definitions into bit-true hardware logic, running **512x faster than real-time** on standard CPUs.

---

## 🚀 Performance Benchmarks (v3.6.0)

| Metric | Legacy Python | **SC-NeuroCore v3.6** | Speedup |
| :--- | :--- | :--- | :--- |
| **LIF Neuron Update** | 12.9 ms | **0.025 ms** | **512.4x** 🚀 |
| **Dense Synaptic Layer** | 64.0 ms | **0.380 ms** | **168.0x** ⚡ |
| **Bit-Stream Encoding** | 51.0 ms | **0.342 ms** | **149.3x** |
| **Inference Latency** | ~2.5 ms | **< 0.010 ms** | **> 250x** |

*Verified against SystemVerilog Hardware Co-Simulation (8/8 Tests Passed).*
[📄 **Read the White Paper**](https://github.com/your-username/sc-neurocore/releases/latest/download/SC_NeuroCore_v3.6_WhitePaper_512x_Benchmarks.pdf)

---

## 📦 Installation

```bash
pip install sc-neurocore
