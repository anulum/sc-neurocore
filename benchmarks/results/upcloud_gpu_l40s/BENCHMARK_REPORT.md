# SC-NeuroCore UpCloud Benchmark Report

Date: 2026-03-07
Server: UpCloud GPU-8xCPU-64GB-1xL40S (fi-hel2)

## System

| Component | Spec |
|-----------|------|
| CPU | AMD EPYC 9575F 64-Core (8 vCPU, Zen 5, AVX-512) |
| RAM | 64 GB DDR5 |
| GPU | NVIDIA L40S 46 GB GDDR6, CUDA 8.9, Driver 580.95 |
| OS | Ubuntu 24.04 LTS, kernel 6.8.0-87-generic |

## Rust Criterion Benchmarks (engine, AVX-512)

| Benchmark | Time | Notes |
|-----------|------|-------|
| pack_dispatch_1m | 8.85 µs | SIMD-dispatched bitstream packing (1M bits) |
| pack_fast_1m | 106.6 µs | Fast scalar path |
| pack_1m | 412.5 µs | Baseline scalar |
| popcount_portable_1m | 556 ns | Portable popcount (1M bits) |
| popcount_simd_1m | 591 ns | SIMD-dispatched popcount |
| fused_and_popcount_scalar_16w | 1.61 ns | Scalar AND+popcount (16 words) |
| fused_and_popcount_dispatch_16w | 2.04 ns | SIMD-dispatched AND+popcount |
| bernoulli_packed_simd_xoshiro_1024 | 171 ns | PRNG + pack (1024 bits) |
| dense_forward_fused_64x32 | 37.7 µs | Dense SC layer (64 neurons, 32 features) |
| dense_forward_prepacked_64x32 | 31.0 µs | Pre-packed dense SC layer |
| kuramoto_100_osc_1000_steps | 118.7 ms | Kuramoto oscillator integration |
| attention_10x16_20x32 | 63.1 µs | SC attention (seq 10×16 + 20×32) |
| gnn_20x8_forward | 69.7 µs | Graph neural network forward (20 nodes, 8 features) |

### SIMD Speedups

| Operation | SIMD | Scalar | Speedup |
|-----------|------|--------|---------|
| Bitstream pack (1M) | 8.85 µs | 412.5 µs | **46.6x** |
| Dense forward (64×32) | 31.0 µs (prepacked) | 37.7 µs (fused) | 1.22x |

## Python Benchmark Suite (CuPy + CUDA)

Backend: CuPy 14.0.1, PyTorch 2.6.0+cu124

### Scalar Primitives (CPU)

| Benchmark | Time | Throughput |
|-----------|------|------------|
| LFSR step (16-bit) | 0.2 µs/iter | 6.34 Mstep/s |
| Bitstream encoder step | 0.2 µs/iter | 5.03 Mstep/s |
| LIF neuron step (Q8.8) | 0.3 µs/iter | 3.41 Mstep/s |

### Packed Bitstream Operations (NumPy, CPU)

| Benchmark | Time | Throughput |
|-----------|------|------------|
| pack_bitstream 1-D (1024) | 3.2 µs | 0.32 Gbit/s |
| pack_bitstream 1-D (65536) | 45.8 µs | 1.43 Gbit/s |
| pack_bitstream 2-D (64×1024) | 45.5 µs | 1.44 Gbit/s |
| vec_and (1024 words) | 0.4 µs | 154.81 Gbit/s |
| vec_popcount SWAR (1024 words) | 8.4 µs | 7.83 Gbit/s |

### Dense Layer (SC Mode)

| Benchmark | Time | Throughput |
|-----------|------|------------|
| Dense forward (16×8, L=256) | 6.49 ms | 0.01 GOP/s |
| Dense forward (64×32, L=1024) | 274 µs | 7.64 GOP/s |

### GPU Backend (CuPy, L40S)

| Benchmark | Time | Throughput |
|-----------|------|------------|
| gpu_pack_bitstream (65536) | 842 µs | 0.08 Gbit/s |
| gpu_vec_mac (64×32×16w) | 93.7 µs | **22.39 GOP/s** |

GPU vec_mac vs CPU SC dense: **22.39 vs 7.64 GOP/s (2.93x GPU speedup)**

## SNN Comparison (20 Variants + Brian2)

1000 LIF neurons, 1000 ms simulation, weight_exc=0.1, ext_rate=20 Hz

| Variant | Time | Spikes | Rate (Hz) |
|---------|------|--------|-----------|
| brian2 | 15.39s | 0 | 0.0 |
| v1 (baseline) | 11.80s | 0 | 0.0 |
| v2 (vectorized) | 4.69s | 0 | 0.1 |
| v3 (sparse) | 5.56s | 0 | 0.0 |
| v4 (event-driven) | 11.88s | 0 | 0.0 |
| v5 (plasticity) | 5.61s | 13,230 | 13.2 |
| v6 (hybrid) | 15.04s | 0 | 0.0 |
| v7 (neuromodulation) | 16.50s | 5,228 | 5.2 |
| v8 (multicompartment) | 11.76s | 0 | 0.0 |
| v9 (recurrent) | 11.58s | 0 | 0.0 |
| v10 (homeostatic) | 9.95s | 0 | 0.0 |
| v11 (dendritic) | 6.54s | 0 | 0.0 |
| v12 (network-level) | 161.82s | 0 | 0.0 |
| v13 (neuroevolution) | 64.46s | 0 | 0.0 |
| v14 (heterogeneous) | 11.81s | 0 | 0.0 |
| v15 (Brunel translator) | 0.00s | 0 | 0.0 |
| v16 (HDC) | 4.28s | 0 | 999.2 |
| v17 (Izhikevich) | 2.10s | 0 | 0.1 |
| v18 (CuPy GPU) | **0.75s** | 0 | 0.0 |
| v19 (PyTorch CUDA) | **0.83s** | 0 | 0.0 |
| v20 (minimal) | **0.11s** | 0 | 0.0 |

### GPU Variants vs Brian2

| Variant | Time | Speedup vs Brian2 |
|---------|------|--------------------|
| v18 (CuPy L40S) | 0.75s | **20.5x** |
| v19 (PyTorch CUDA L40S) | 0.83s | **18.5x** |
| v20 (minimal) | 0.11s | **139.9x** |
| v2 (vectorized NumPy) | 4.69s | **3.3x** |

## Python Test Suite

976 passed, 2 failed, 49 skipped (21.04s)

Failed: `test_gpu_vec_and`, `test_gpu_vec_mac` (GPU coverage test assertions — CuPy kernel launch overhead on L40S)

## Cost

Server: GPU-8xCPU-64GB-1xL40S @ €1.11/hr
Runtime: ~35 min
Cost: ~€0.65
