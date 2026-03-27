# Cross-Framework SNN Benchmark

Balanced E-I LIF network (1,000 neurons, 80/20 split, 10% random
connectivity, Poisson external drive, 300ms simulation, dt=0.1ms).

Measured on Intel i5-11600K @ 3.90 GHz, Python 3.12.10, Windows 11.
Single-threaded. No GPU.

## Results (1,000 neurons)

| Framework | Mode | Time (s) | Peak Memory | Speedup vs Brian2 |
|-----------|------|----------|-------------|-------------------|
| **SC-NeuroCore** | **Rust engine** | **0.08** | **0.3 MB** | **835×** |
| SC-NeuroCore | NumPy | 12.9 | 26.4 MB | 5.2× |
| Norse | PyTorch CPU | 41.7 | 18.9 MB | 1.6× |
| snnTorch | PyTorch CPU | 49.1 | 48.5 MB | 1.4× |
| Brian2 | runtime (NumPy) | 67.7 | 87.9 MB | 1.0× (baseline) |

### Caveats

- Brian2 "runtime" mode uses NumPy codegen, not Brian2's fastest path.
  Brian2 C++ standalone compiles the network to native C++ and would be
  significantly faster. We have not yet measured standalone mode on this
  machine (compilation time is excluded from Brian2's reported time in
  standalone mode, which affects fair comparison).
- snnTorch and Norse step through the network in PyTorch without
  compiled graph optimisation (`torch.compile` was not used).
- SC-NeuroCore Rust uses a fused E-I simulation in compiled Rust with
  CSR connectivity, Poisson input, and Euler integration in a single
  FFI call.
- SC-NeuroCore NumPy uses pure Python + NumPy vectorised operations
  with the same algorithm.
- Low spike counts indicate the external drive was suboptimal for this
  network configuration. All frameworks received the same input
  parameters, so timing comparisons remain valid (all did equivalent
  computational work).

### What This Means

The Rust engine's advantage comes from:

1. **Zero Python overhead per timestep** — the entire simulation loop
   runs in compiled Rust
2. **CSR sparse connectivity** — only non-zero synapses are stored and
   iterated
3. **Fused kernels** — connectivity build + Poisson input + Euler step +
   spike detection in one call, no intermediate allocations

The NumPy backend is already competitive with snnTorch and Norse despite
being pure Python, because the algorithm is vectorised over neurons
rather than using per-neuron Python loops.

## Reproduce

```bash
pip install sc-neurocore sc-neurocore-engine brian2 snntorch norse
python benchmarks/cross_framework_benchmark.py --scales 1000 --json results.json
```

Stored artifact:
[`benchmarks/results/cross_framework_1k.json`](https://github.com/anulum/sc-neurocore/blob/main/benchmarks/results/cross_framework_1k.json)

## Benchmark Script

Source: [`benchmarks/cross_framework_benchmark.py`](https://github.com/anulum/sc-neurocore/blob/main/benchmarks/cross_framework_benchmark.py)

Tests all frameworks with the same network topology, parameters, and
simulation duration. Measures wall time via `time.perf_counter()` and
peak memory via `tracemalloc`.
