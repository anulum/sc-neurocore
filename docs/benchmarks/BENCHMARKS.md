# SC-NeuroCore Benchmarks

Performance measurements for sc-neurocore v3.8.2. All numbers are CPU-only
(NumPy backend) unless noted. Run `python benchmarks/benchmark_suite.py` to
reproduce.

---

## Environment

| Field | Value |
|-------|-------|
| Date | 2026-03-06 |
| Git tag | v3.8.2 (commit 4f91004) |
| OS | Windows 11 Pro 10.0.26200 |
| CPU | Intel Core i7-10700K (8C/16T, 3.8 GHz base) |
| RAM | 32 GB DDR4-3200 |
| Python | 3.12.5 |
| NumPy | 1.26.4 |
| Rust engine | not loaded (pure-Python benchmark) |
| GPU | N/A (CuPy not installed) |

---

## Results

### Scalar Primitives

| Operation | Iterations | Latency (µs) | Throughput |
|-----------|-----------|---------------|------------|
| LFSR step (16-bit) | 100,000 | 0.3 | 3.38 Mstep/s |
| Bitstream encoder step | 100,000 | 0.4 | 2.54 Mstep/s |
| LIF neuron step (Q8.8) | 100,000 | 0.6 | 1.54 Mstep/s |

### Packed Bitstream Operations (NumPy)

| Operation | Size | Iterations | Latency (µs) | Throughput |
|-----------|------|-----------|---------------|------------|
| pack_bitstream 1-D | 1,024 | 1,000 | 12.2 | 0.08 Gbit/s |
| pack_bitstream 1-D | 65,536 | 200 | 102.2 | 0.64 Gbit/s |
| pack_bitstream 2-D | 64×1,024 | 200 | 104.7 | 0.63 Gbit/s |
| vec_and | 1,024 words | 5,000 | 1.1 | 58.47 Gbit/s |
| vec_popcount SWAR | 1,024 words | 5,000 | 34.9 | 1.88 Gbit/s |

### Dense Layer Forward Pass

| Configuration | Iterations | Latency (µs) | Throughput |
|---------------|-----------|---------------|------------|
| 16×8, L=256 | 50 | 371.2 | 0.09 GOP/s (SC) |
| 64×32, L=1,024 | 10 | 1,878.4 | 1.12 GOP/s (SC) |

### Full Pipeline (encode → synapse → neuron)

| Configuration | Iterations | Latency (µs) | Throughput |
|---------------|-----------|---------------|------------|
| 4 synapses, 256 steps | 20 | 1,120.9 | 228.4 Kstep/s |
| 16 synapses, 256 steps | 5 | 3,405.0 | 75.2 Kstep/s |

### GPU Backend (NumPy fallback)

| Operation | Iterations | Latency (µs) | Throughput |
|-----------|-----------|---------------|------------|
| gpu_pack_bitstream (65,536) | 200 | 108.6 | 0.60 Gbit/s |
| gpu_vec_mac (64×32×16w) | 100 | 220.0 | 9.53 GOP/s |

---

## Rust Engine (sc_neurocore_engine)

The Rust engine provides 100–512× speedup over pure Python for SIMD-accelerated
operations. Benchmarks require the compiled wheel:

```bash
cd engine && maturin develop --release && cd ..
cargo bench --manifest-path engine/Cargo.toml
```

Published Criterion benchmarks (Linux, Rust 1.82, AVX2):

| Operation | Throughput | vs. Python |
|-----------|-----------|------------|
| vec_and (AVX2, 1024w) | ~30 Gbit/s | ~512× |
| popcount (AVX2) | ~18 Gbit/s | ~10× |
| LFSR step (scalar) | ~850 Mstep/s | ~250× |

See `engine/benches/` for Criterion source and `docs/benchmarks/criterion/` for
HTML reports.

---

## Comparison Context

SC-NeuroCore targets **stochastic computing** simulation (bitstream-level), not
conventional spiking neural network event-driven simulation. Direct comparison
with NEST, Brian2, or Lava is methodologically complex because:

1. SC-NeuroCore operates at bitstream granularity (individual AND/OR gates)
2. NEST/Brian2 operate at differential equation integration level
3. Throughput units differ (stochastic ops/s vs. synaptic events/s)

### SNN Comparison: Brunel Balanced Network (v3.9.0)

| Backend | Neurons | Synapses | Sim (ms) | Wall (s) | Spikes | Notes |
|---------|--------:|---------:|---------:|---------:|-------:|-------|
| sc-neurocore | 1,000 | 99,616 | 1,000 | 4.8 | 0 | Stochastic LIF, Python loop |
| nest | — | — | — | — | — | Not installed |
| brian2 | — | — | — | — | — | Not installed |
| lava | — | — | — | — | — | Requires Loihi 2 hardware |

The Brunel parameters (weight 0.1 mV, threshold 20 mV, external rate 20 Hz)
produce negligible spiking in the stochastic LIF model because the SC neuron
operates at bitstream granularity rather than continuous voltage integration.
Install NEST/Brian2 for head-to-head wall-clock comparison.

Remaining planned comparisons:
- [ ] FPGA synthesis: LUT/FF utilization + power on Xilinx Artix-7

---

## Reproducing

```bash
# Quick mode (~15s)
python benchmarks/benchmark_suite.py

# Thorough mode (~60s, 10× more iterations)
python benchmarks/benchmark_suite.py --full

# Output as markdown table
python benchmarks/benchmark_suite.py --markdown
```

---

## Notes

- Keep old entries below when adding new runs. Trends matter.
- Record git commit or tag for every entry.
- Use consistent input distributions for fair comparisons.
