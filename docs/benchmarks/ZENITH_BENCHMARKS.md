# Project Zenith: End-to-End Benchmark Report

This document contains automated hardware profiling results mapping the `autonomous_learning` Zenith bridge overhead limits.
Measurements are recorded precisely matching physical memory limits against CPU arrays simulating multi-threaded workloads for physical hardware design scaling (Exa-scale architectures).

## How to Reproduce on Your Hardware

To execute this benchmark matrix locally and evaluate your own hardware architectures limits:
```bash
python benchmarks/bench_zenith_e2e.py --scale 100000 1000000 10000000 --rules all
```
*Note: Default scale is `1_000_000` nodes for quick local runs. Use `--scale 10000000` for true exascale previews (requires ≥ 16 GB RAM).*

## Hardware & Environment Overview
- **System Memory:** 31.1 GB RAM
- **Logical CPU Cores:** 12
- **PyTorch Device:** `cpu` (Fallback from unsupported CUDA architecture)
- **Scale:** `10_000_000` Nodes (Approx **~152 - 228 MB** memory array buffers natively)

> [!WARNING]
> **Performance Note**: The Torch numbers below reflect the CPU fallback (no CUDA) which utilizes highly-optimized MKL multithreading internally. On GPU, Torch forward/backward would be significantly faster (expected 5-10X faster). GPU Torch results will be added in a follow-up benchmark.
> The Rust numbers represent purely CPU-bound Rayon multithreaded bounds tracking exact deterministic physics overheads out of process.

## Parity Bound Verifications
Physical hardware architectures require symmetric execution constraints across framework FFI boundaries explicitly bypassing floating-point errors.
- **Formal Torch/Rust Step Parity (Deterministic)**: ✓ `PASS` *(Tolerance: 1e-6)*
- **Rust Exascale `.scal` Load Identity**: ✓ `PASS` *(Tolerance: 1e-6)*

### 1. PyTorch Surrogate Autograd Throughput
Measuring backpropagation graph traversals mimicking ML scaling frameworks natively out-of-place.
| Rule | Forward Pass | Backward Proxy Map | Total Backprop Loop |
|------|--------------|--------------------|---------------------|
| **ELIGENT** | `181.79 ms` | `45.91 ms` | **227.70 ms** |
| **STDP** | `143.20 ms` | `63.41 ms` | **206.61 ms** |
| **R-STDP** | `170.60 ms` | `102.59 ms` | **273.20 ms** |
| **BCM** | `186.66 ms` | `165.58 ms` | **352.24 ms** |

*Note: The hardware fallbacked to multi-threaded CPU tensors on the test machine.*

### 2. Rust Native Physics Deterministic Limits
Measures the bare-metal Rayon simulation limit passing strictly boolean physical layouts.
| Rule | Deterministic Parallel `step_analog` | Memory Topology Size |
|------|--------------------------------------|----------------------|
| **ELIGENT** | `697.46 ms` | 190.73 MB |
| **STDP** | `696.68 ms` | 190.73 MB |
| **R-STDP** | `713.18 ms` | 228.88 MB |
| **BCM** | `705.27 ms` | 152.58 MB |

### 3. Exascale Persistence Protocol (`.scal` IO)
Demonstrating direct `mmap`-ready memory write/reads bypassing the Python `pickle` Global Interpreter Lock bounds directly across OS storage arrays.

| Rule | File Size | Disk Write Speed | Dump Latency | Disk Read Speed | Load Latency |
|------|-----------|------------------|--------------|-----------------|--------------|
| **ELIGENT** | 190.73 MB | `284.66 MB/s` | `670.02 ms` | `1494.53 MB/s`| **`127.62 ms`** |
| **STDP** | 190.73 MB | `284.57 MB/s` | `670.24 ms` | `1341.15 MB/s`| **`142.21 ms`** |
| **R-STDP** | 228.88 MB | `241.25 MB/s` | `948.71 ms` | `1024.18 MB/s`| **`223.47 ms`** |
| **BCM** | 152.58 MB | `228.06 MB/s` | `669.04 ms` | `1169.25 MB/s`| **`130.49 ms`** |

## See Also
For extremely specific and parallel GPU scaling limits bridging `autonomous_learning` across PyTorch and pure-Rust edge deployment bindings natively via WGSL, see the [End-to-End GPU Benchmark Report](ZENITH_BENCHMARKS_GPU.md).
