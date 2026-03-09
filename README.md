© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AFFERO GENERAL PUBLIC LICENSE v3
Commercial Licensing: Available

# SC-NeuroCore

<p align="center">
  <img src="docs/assets/sc_neurocore_header.png" width="1280" alt="SC-NeuroCore — Stochastic Computing & Neuromorphic Engine">
</p>

[![CI](https://github.com/anulum/sc-neurocore/actions/workflows/ci.yml/badge.svg)](https://github.com/anulum/sc-neurocore/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/sc-neurocore)](https://pypi.org/project/sc-neurocore/)
[![crates.io](https://img.shields.io/crates/v/sc_neurocore_engine)](https://crates.io/crates/sc_neurocore_engine)
[![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen)](https://github.com/anulum/sc-neurocore)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18594898-blue)](https://doi.org/10.5281/zenodo.18594898)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/anulum/sc-neurocore/badge)](https://scorecard.dev/viewer/?uri=github.com/anulum/sc-neurocore)

Design spiking neural networks in Python, simulate them bit-exactly, and compile to FPGA — using stochastic computing, where an AND gate is a multiplier and a wire is a number.

```bash
pip install sc-neurocore
```

## What it does

```python
from sc_neurocore import SCDenseLayer, VectorizedSCLayer
from sc_neurocore.hdl_gen import VerilogGenerator

# 1. Simulate in Python (bit-true digital twin)
layer = VectorizedSCLayer(n_inputs=8, n_neurons=4, length=1024)
output = layer.forward(input_probs)   # stochastic bitstream computation

# 2. Generate synthesisable Verilog from the same architecture
gen = VerilogGenerator(module_name="my_snn")
gen.add_layer("Dense", "hidden", {"n_neurons": 16})
gen.add_layer("Dense", "output", {"n_neurons": 4})
verilog = gen.generate()              # → sc_dense_layer_core + AXI-Lite wrapper
```

The Python model and Verilog RTL use identical LFSR seeds, Q8.8 fixed-point
arithmetic, and overflow semantics — what you simulate is what you synthesise.

## Architecture

```
Python API ──→ Rust Engine (AVX-512/NEON) ──→ IR Compiler ──→ Verilog RTL ──→ FPGA
   │                                                              │
   └── bit-true simulation (digital twin) ◄── co-sim check ──────┘
```

**Three acceleration paths**: NumPy (pure Python), Rust SIMD (`sc_neurocore_engine`), or CuPy GPU.

## Benchmarks

### Rust SIMD Engine (AVX-512, Criterion)

| Operation | Throughput | vs Python |
|-----------|-----------|----------|
| Bitstream packing (`pack_dispatch`) | 41.3 Gbit/s | 79× |
| Popcount (`VPOPCNTDQ`) | 366 Mword/s | 10.6× |
| Fused encode+AND+popcount (Xoshiro) | 285 ns / 1024 bits | — |
| Dense forward 128→64, L=1024 | — | 7.3× |
| Dense forward prepacked 64→32 | 54.9 µs | 43.8× |

### Brian2 Comparison — Brunel Balanced Network

1000 LIF neurons (80E/20I), 10% connectivity, 1000 ms. Delta-PSC semantics
(`v += w`), Poisson drive at 200 Hz.

| Backend | Wall time | Speedup vs V1 |
|---------|-------:|------:|
| V1 per-neuron Python | 49.3 s | 1.0× |
| V20 vectorized NumPy | 10.3 s | 4.8× |
| V18 Numba JIT | 5.2 s | 9.5× |
| V19 PyTorch CUDA (GTX 1060) | 5.7 s | 8.7× |
| Brian2 (Cython codegen) | 1.6 s | 30.8× |

Brian2 is faster at this scale because its C++ codegen + sparse synapse
representation amortises well above ~1K neurons. SC-NeuroCore targets
FPGA-scale networks (≤1K neurons) where bit-exact RTL co-simulation matters
and Brian2 has no hardware path. At 1K neurons on cloud hardware (EPYC 9575F),
SC dense operations complete in 55 ms vs Brian2's 6.2 s first-run (114×);
the gap narrows on subsequent Brian2 runs after Cython compilation.

Full 20-variant translator results and cloud scaling data:
[docs/benchmarks/BENCHMARKS.md](docs/benchmarks/BENCHMARKS.md)

### FPGA Synthesis (Yosys, Xilinx 7-series)

| Module | LUTs | FFs |
|--------|-----:|----:|
| `sc_neurocore_top` (3-in, 7-neuron) | 7,382 | 2,442 |

MNIST classifier (16→10, PCA) estimated at ~56K LUTs — fits Artix-7 100T.
See [docs/tutorials/fpga_in_20_minutes.md](docs/tutorials/fpga_in_20_minutes.md).

## Hardware (Verilog RTL)

Nine synthesisable modules in `hdl/`:
- `sc_bitstream_encoder.v` — LFSR-based stochastic encoder (Q8.8 comparator)
- `sc_bitstream_synapse.v` — AND-gate multiplier (1 LUT)
- `sc_lif_neuron.v` — Q8.8 leaky integrate-and-fire
- `sc_dense_layer_core.v` — Full pipeline with decorrelated seeds
- `sc_dense_matrix_layer.v` — Per-neuron weight dense layer (MNIST-scale)
- `sc_neurocore_top.v` — AXI-Lite configuration wrapper

Co-simulation verifies bit-exact equivalence:
```bash
python scripts/cosim_gen_and_check.py --generate
iverilog -o tb_lif hdl/sc_lif_neuron.v hdl/tb_sc_lif_neuron.v && vvp tb_lif
python scripts/cosim_gen_and_check.py --check
```

### MNIST-on-FPGA Demo

Train a digit classifier, quantise to Q8.8, simulate with stochastic
bitstreams (bit-exact match to RTL), and export Verilog weights:

```bash
python examples/mnist_fpga/demo.py
python examples/mnist_fpga/demo.py --export-verilog hdl/generated/mnist_weights.vh
```

Vivado timing/power analysis (requires Vivado):
```bash
vivado -mode batch -source tools/vivado_impl.tcl -tclargs -top sc_dense_matrix_layer -part xc7a100tcsg324-1
python tools/vivado_report.py vivado_reports/
```

## Documentation

**[anulum.github.io/sc-neurocore](https://anulum.github.io/sc-neurocore/)** — full docs, API reference, hardware guide, benchmarks.

| Resource | Link |
|----------|------|
| Getting Started | [docs/guides/getting-started.md](docs/guides/getting-started.md) |
| API Reference | [docs/api/API_REFERENCE.md](docs/api/API_REFERENCE.md) |
| Hardware Guide | [docs/hardware/HARDWARE_GUIDE.md](docs/hardware/HARDWARE_GUIDE.md) |
| Benchmarks | [docs/benchmarks/BENCHMARKS.md](docs/benchmarks/BENCHMARKS.md) |
| Examples | [examples/](examples/) (11 runnable scripts) |
| Changelog | [CHANGELOG.md](CHANGELOG.md) |

## Install extras

```bash
pip install sc-neurocore[gpu]      # CuPy CUDA acceleration
pip install sc-neurocore[quantum]  # Qiskit + PennyLane bridges
pip install sc-neurocore[full]     # everything
pip install -e ".[dev]"            # development (all modules + test tools)
```

## Community

- [GitHub Discussions](https://github.com/anulum/sc-neurocore/discussions)
- [Issue Tracker](https://github.com/anulum/sc-neurocore/issues)
- [Contributing Guide](CONTRIBUTING.md)

## License

Dual-licensed: [AGPLv3](LICENSE) (open source) or commercial license.
Contact [protoscience@anulum.li](mailto:protoscience@anulum.li) for commercial enquiries.
