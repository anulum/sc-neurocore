# SC-NeuroCore v3.7

[![CI](https://github.com/anulum/sc-neurocore/actions/workflows/v3-engine.yml/badge.svg)](https://github.com/anulum/sc-neurocore/actions)
[![PyPI](https://img.shields.io/pypi/v/sc-neurocore.svg)](https://pypi.org/project/sc-neurocore/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18594898.svg)](https://doi.org/10.5281/zenodo.18594898)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Hardware: Verified](https://img.shields.io/badge/Hardware-Verified_Bit--True-green)](cosim/)

**The Industry's First Verified Rust-Based Neuromorphic Compiler.**

> SC-NeuroCore translates high-level Python SNN definitions into bit-true hardware logic, running **512x faster than real-time** on standard CPUs. v3.7 extends the stochastic kernel into a **polymorphic engine** supporting HDC/VSA, Petri Nets, and fault-tolerant binary streams.

---

## Quick Start

```bash
pip install sc-neurocore          # PyPI wheel (Linux/macOS/Windows)
# — or build from source —
cd engine && maturin develop --release
```

### SNN Dense Forward

```python
import numpy as np
from sc_neurocore_engine import DenseLayer

layer = DenseLayer(n_inputs=784, n_neurons=128, length=1024)
x = np.random.rand(784)
y = layer.forward_numpy(x)           # single sample — returns (128,)
batch = np.random.rand(50, 784)
Y = layer.forward_batch_numpy(batch)  # 50 samples in one FFI call — (50, 128)
```

### Hyper-Dimensional Computing (HDC/VSA)

```python
from sc_neurocore_engine import HDCVector

# Encode knowledge: country–capital pairs in 10,000-bit hypervectors
role_country = HDCVector(10_000)
role_capital = HDCVector(10_000)

france  = HDCVector(10_000)
paris   = HDCVector(10_000)
germany = HDCVector(10_000)
berlin  = HDCVector(10_000)

record_fr = (france * role_country) + (paris   * role_capital)  # bind + bundle
record_de = (germany * role_country) + (berlin  * role_capital)

# Query: "Capital of France?"
query = record_fr * role_capital      # unbind capital role
print(paris.similarity(query))        # ~0.62  (highest match)
print(berlin.similarity(query))       # ~0.50  (near chance)
```

### Stochastic Petri Net

```python
import numpy as np
from sc_neurocore_engine import PetriNetEngine

w_in  = np.array([[1, 0], [0, 1], [0, 0]], dtype=np.float64)  # 3 places, 2 transitions
w_out = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
thresholds = np.array([0.5, 0.5])

engine = PetriNetEngine({"w_in": w_in, "w_out": w_out, "thresholds": thresholds})
for _ in range(10):
    engine.step()
print(engine.marking)  # token distribution after 10 firing cycles
```

### Fault-Tolerant Boolean Logic

```python
from sc_neurocore_engine import BitStreamTensor

# Encode TRUE as 1024 ones, inject 5% noise — majority-vote decoding recovers truth
t = BitStreamTensor.from_packed([0xFFFFFFFFFFFFFFFF] * 16, 1024)
print(t.popcount())  # 1024 — all ones (TRUE)
# After noise injection and AND/OR/NOT, majority vote still decodes correctly
```

### LIF Neuron Simulation

```python
from sc_neurocore_engine import batch_lif_run

spikes, voltages = batch_lif_run(
    n_steps=10_000,
    leak_k=200, gain_k=256,
    i_t=300,
    v_threshold=256,
)
print(f"Spike count: {spikes.sum()}")
```

### IR Compilation to SystemVerilog

```python
from sc_neurocore_engine import ScGraphBuilder, ir_verify, ir_emit_sv

b = ScGraphBuilder("my_net")
inp = b.input("x", "rate")
enc = b.encode(inp, length=1024, seed=0xACE1)
pc  = b.popcount(enc)
b.output("y", pc)
graph = b.build()

errors = ir_verify(graph)
assert errors is None, errors

sv_code = ir_emit_sv(graph)
print(sv_code[:200])  # synthesisable SystemVerilog
```

---

## Domain-Specific Profiles

SC-NeuroCore is a **polymorphic engine**. The core stochastic kernel supports specialised modes:

### 1. SCPN Profile (Industrial Control)
* **Use Case:** Verification of asynchronous logic and Petri Nets.
* **Feature:** Maps Places and Transitions to stochastic matrices via `PetriNetEngine`.
* **Status:** **Included** (See [`bridge/sc_neurocore_engine/petri_net.py`](bridge/sc_neurocore_engine/petri_net.py) and [`examples/04_scpn_stack.py`](examples/04_scpn_stack.py)).

### 2. HDC Profile (Symbolic AI)
* **Use Case:** Hyper-Dimensional Computing for low-power sensor fusion and symbolic reasoning.
* **Feature:** 10,000-bit vector algebra (Bind, Bundle, Permute) with AVX-512/AVX2 SIMD acceleration.
* **Status:** **Included** (See [`examples/06_hdc_symbolic_query.py`](examples/06_hdc_symbolic_query.py)).

### 3. Bio-Hybrid Interface (R&D)
* **Use Case:** Modelling Gene Regulatory Networks and DNA Storage dynamics.
* **Feature:** Non-linear chemical reaction kinetics via stochastic bitstreams.
* **Status:** **Experimental** (Research licence required for commercial biotech use).

---

## Performance Benchmarks (v3.7.0)

| Metric | Legacy Python | **SC-NeuroCore v3.7** | Speedup |
| :--- | :--- | :--- | :--- |
| **LIF Neuron Update** | 12.9 ms | **0.025 ms** | **512.4x** |
| **Dense Synaptic Layer** | 64.0 ms | **0.380 ms** | **168.0x** |
| **Bit-Stream Encoding** | 51.0 ms | **0.342 ms** | **149.3x** |
| **Inference Latency** | ~2.5 ms | **< 0.010 ms** | **> 250x** |

*Verified against SystemVerilog Hardware Co-Simulation (8/8 Tests Passed).*

[Read the White Paper (PDF)](SC_NeuroCore_v3.6_WhitePaper_512x_Benchmarks.pdf)

---

## Architecture

```
Python API (sc_neurocore_engine)
    |
    +-- HDCVector          — 10k-bit hypervectors (bind/bundle/permute/similarity)
    +-- PetriNetEngine     — Stochastic Colored Petri Net via DenseLayer
    +-- DenseLayer         — SNN dense forward (single, fast, prepacked, batch)
    +-- FixedPointLif      — Leaky Integrate-and-Fire neuron (fixed-point)
    +-- SurrogateLif       — Differentiable LIF (FastSigmoid, SuperSpike, ArcTan)
    +-- StochasticAttention — Multi-head SC attention
    +-- KuramotoSolver     — Phase oscillator (Kuramoto + SSGF geometry coupling)
    +-- ScGraphBuilder     — IR graph -> SystemVerilog compiler
    |
    v
Rust Engine (PyO3 + maturin)
    |
    +-- bitstream.rs       — BitStreamTensor: pack/unpack/XOR/rotate/hamming/bundle
    +-- simd/              — AVX-512 / AVX2 / NEON / portable dispatch
    +-- layer.rs           — Fused encode+AND+popcount dense pipeline
    +-- neuron.rs          — Fixed-point LIF with branchless SIMD
    +-- ir/                — 11 op types -> SystemVerilog emitter
    +-- scpn/              — Kuramoto solver + SSGF + SCPN metrics
```

### SIMD Dispatch

Every hot path auto-selects the fastest available instruction set at runtime:

| Tier | Instructions | Detected via |
|------|-------------|-------------|
| AVX-512 VPOPCNTDQ | fused AND/XOR + popcount in 512-bit lanes | `is_x86_feature_detected!` |
| AVX-512BW | Bernoulli threshold compare, pack | `is_x86_feature_detected!` |
| AVX2 | 256-bit AND/XOR + LUT popcount | `is_x86_feature_detected!` |
| NEON | 128-bit popcount (aarch64 baseline) | compile-time `cfg` |
| Portable | `u64::count_ones()` scalar fallback | always available |

Check your tier: `sc_neurocore_engine.simd_tier()`

---

## API Reference

### Core Classes

| Class | Purpose | Key methods |
|-------|---------|-------------|
| `DenseLayer(n_in, n_out, length, seed)` | SNN dense layer | `forward_numpy()`, `forward_batch_numpy()`, `forward_prepacked_numpy()` |
| `BitStreamTensor(dim, seed)` | Packed bitstream (HDC primitive) | `xor()`, `rotate_right()`, `hamming_distance()`, `bundle()`, `popcount()` |
| `HDCVector(dim, seed)` | High-level HDC vector | `*` (bind), `+` (bundle), `.similarity()`, `.permute()` |
| `PetriNetEngine(artifacts)` | Stochastic Petri Net | `step()`, `run()`, `reset()`, `.marking` |
| `FixedPointLif(...)` | LIF neuron | `step()`, `reset()`, `get_state()` |
| `SurrogateLif(...)` | Differentiable LIF | `forward()`, `backward()` |
| `DifferentiableDenseLayer(...)` | Trainable dense | `forward()`, `backward()`, `update_weights()` |
| `StochasticAttention(dim_k)` | SC attention | `forward()`, `forward_sc()`, `forward_multihead()` |
| `StochasticGraphLayer(adj, n_feat)` | SC graph layer | `forward()`, `forward_sc()` |
| `KuramotoSolver(omega, K, phases)` | Phase oscillator | `step()`, `run()`, `step_ssgf()`, `.phases` |
| `ScGraphBuilder(name)` | IR compiler | `input()`, `encode()`, `popcount()`, `build()` |

### Batch Functions

| Function | Description |
|----------|-------------|
| `batch_lif_run(n_steps, ...)` | Single-neuron LIF simulation |
| `batch_lif_run_multi(n_neurons, n_steps, ...)` | Parallel multi-neuron (Rayon) |
| `batch_lif_run_varying(leak, gain, currents, noises)` | Per-step varying input |
| `batch_encode(probs, length, seed)` | Bernoulli -> packed `u64` |
| `batch_encode_numpy(probs, length, seed)` | Zero-copy NumPy variant |

### Bitstream Utilities

| Function | Description |
|----------|-------------|
| `pack_bitstream(bits)` | `u8[]` -> packed `u64[]` |
| `unpack_bitstream(packed, length)` | packed `u64[]` -> `u8[]` |
| `pack_bitstream_numpy(bits)` | Zero-copy NumPy pack |
| `unpack_bitstream_numpy(packed, length)` | Zero-copy NumPy unpack |
| `popcount(packed)` / `popcount_numpy(packed)` | Count set bits |
| `simd_tier()` | Report SIMD capability |
| `set_num_threads(n)` | Configure Rayon thread pool |

### IR Compilation

| Function | Description |
|----------|-------------|
| `ir_verify(graph)` | Validate IR graph |
| `ir_print(graph)` | Human-readable IR text |
| `ir_parse(text)` | Parse IR from text |
| `ir_emit_sv(graph)` | Emit synthesisable SystemVerilog |

---

## Test Suite

```bash
# Rust (90 tests including 12 property-based)
cd engine && cargo test

# Python (49 tests)
python -m pytest tests/ -v

# Hardware co-simulation (requires Verilator)
python -m pytest cosim/ -v
```

| Category | Count | Coverage |
|----------|-------|----------|
| Rust unit + integration | 75 | bitstream, LIF, dense, SIMD, IR, Kuramoto |
| Rust proptest (fuzzing) | 12 | bitstream roundtrip, neuron bounds, layer invariants |
| Rust HDC integration | 15 | XOR truth tables, rotate, hamming, bundle |
| Python HDC | 20 | BitStreamTensor + HDCVector end-to-end |
| Python Petri Net | 9 | init, step, run, reset, conservation |
| Python bridge | 20 | existing v3.0-v3.6 API |
| Co-simulation | 8 | Verilator vs Rust golden model |

---

## Examples

| File | Description |
|------|-------------|
| [`01_basic_sc_encoding.py`](examples/01_basic_sc_encoding.py) | Bernoulli bitstream encoding |
| [`02_ir_compile_demo.py`](examples/02_ir_compile_demo.py) | IR graph -> SystemVerilog pipeline |
| [`03_benchmark_report.py`](examples/03_benchmark_report.py) | Performance benchmarks |
| [`04_scpn_stack.py`](examples/04_scpn_stack.py) | SCPN Kuramoto oscillator stack |
| [`05_hdl_generation.py`](examples/05_hdl_generation.py) | HDL code generation |
| [`06_hdc_symbolic_query.py`](examples/06_hdc_symbolic_query.py) | HDC "Capital of France?" demo |
| [`07_safety_critical_logic.py`](examples/07_safety_critical_logic.py) | Fault-tolerant Boolean logic |

---

## Building from Source

**Prerequisites:** Rust 1.75+ toolchain, Python 3.9-3.12, maturin

```bash
git clone https://github.com/anulum/sc-neurocore.git
cd sc-neurocore/engine
maturin develop --release       # builds + installs into current venv
cargo test                      # run Rust tests
cd .. && python -m pytest tests/ -v  # run Python tests
```

**Cross-platform wheels** are built via CI for Linux, macOS, and Windows.

---

## Citation

```bibtex
@software{sotek_sc_neurocore_2026,
  author    = {Šotek, Miroslav},
  title     = {SC-NeuroCore v3.7: 512x Real-Time Stochastic Neuromorphic Compiler},
  version   = {3.7.0},
  doi       = {10.5281/zenodo.18594898},
  url       = {https://github.com/anulum/sc-neurocore},
  year      = {2026},
  license   = {AGPL-3.0}
}
```

---

## Licence

[GNU Affero General Public License v3.0](LICENSE) — Commercial licensing available.
Contact: protoscience@anulum.li | [anulum.li](https://www.anulum.li)
