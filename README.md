# SC-NeuroCore

**Version:** 2.1.0
**Status:** Production Core Verified | 460 Tests Passing | CI/CD Active

SC-NeuroCore is a universal stochastic computing framework for neuromorphic
hardware simulation. It provides bit-true Python models that match Verilog RTL
cycle-exactly, GPU-accelerated inference, and a tiered module system spanning
production hardware to theoretical research.

## Quick Start

```bash
# Install core package
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run benchmarks
python scripts/benchmark_suite.py

# GPU acceleration (requires CUDA)
pip install -e ".[gpu]"
```

## Architecture

### Module Tiers

| Tier | Subpackages | Description |
|------|-------------|-------------|
| **core** | neurons, synapses, layers, sources, utils, recorders, accel | Production-ready. Imported by default. |
| **research** | hdc, solvers, transformers, quantum, robotics, bio, physics, +18 more | Functional but experimental. Import explicitly. |
| **contrib** | exotic, meta, transcendent, eschaton, post_silicon | Speculative / theoretical. Import explicitly. |

### Core API (28 symbols)

```python
from sc_neurocore import (
    # Neurons
    StochasticLIFNeuron, FixedPointLIFNeuron, FixedPointLFSR,
    FixedPointBitstreamEncoder, HomeostaticLIFNeuron,
    StochasticDendriticNeuron, SCIzhikevichNeuron,
    # Synapses
    BitstreamSynapse, BitstreamDotProduct,
    StochasticSTDPSynapse, RewardModulatedSTDPSynapse,
    # Layers
    SCDenseLayer, SCConv2DLayer, SCLearningLayer,
    VectorizedSCLayer, SCRecurrentLayer, MemristiveDenseLayer,
    SCFusionLayer, StochasticAttention,
    # Utilities
    BitstreamEncoder, BitstreamAverager, RNG,
    generate_bernoulli_bitstream, generate_sobol_bitstream,
    bitstream_to_probability,
    # Sources & Recorders
    BitstreamCurrentSource, BitstreamSpikeRecorder,
)
```

### Hardware (Verilog RTL)

```
hdl/
  sc_bitstream_encoder.v   -- LFSR-based stochastic encoder (SEED_INIT param)
  sc_bitstream_synapse.v   -- AND-gate SC multiplier
  sc_dotproduct_to_current.v -- Popcount -> fixed-point current
  sc_lif_neuron.v          -- Q8.8 leaky integrate-and-fire
  sc_firing_rate_bank.v    -- Spike rate estimator
  sc_dense_layer_core.v    -- Full dense layer pipeline (decorrelated seeds)
  sc_neurocore_top.v       -- AXI-Lite configuration wrapper
  sc_axil_cfg.v            -- AXI-Lite register file
  tb_sc_lif_neuron.v       -- Co-simulation testbench
```

### GPU Acceleration

```python
from sc_neurocore.accel import xp, HAS_CUPY, to_device, to_host
from sc_neurocore.accel.gpu_backend import gpu_vec_mac

# VectorizedSCLayer auto-detects GPU
layer = VectorizedSCLayer(n_inputs=32, n_neurons=64, length=1024)
output = layer.forward(input_values)  # GPU if CuPy available, else CPU
```

## Hardware-Software Co-Simulation

The co-sim flow verifies bit-exact equivalence between the Python model and
Verilog RTL:

```bash
# 1. Generate stimuli + expected results (Python golden model)
python scripts/cosim_gen_and_check.py --generate

# 2. Run Verilog simulation (requires Icarus Verilog)
iverilog -o tb_lif hdl/sc_lif_neuron.v hdl/tb_sc_lif_neuron.v
vvp tb_lif

# 3. Compare results
python scripts/cosim_gen_and_check.py --check
```

### Key Technical Details

- **LFSR**: 16-bit maximal-length, polynomial x^16+x^14+x^13+x^11+1, period 65535
- **Seed strategy**: Input encoders `0xACE1 + i*7`, weight encoders `0xBEEF + i*13`
- **Fixed-point**: Q8.8 (DATA_WIDTH=16, FRACTION=8), signed two's complement
- **Overflow**: Explicit bit-width masking via `_mask()` function

## CI/CD

GitHub Actions pipeline (`.github/workflows/sc-neurocore-ci.yml`):
- **Lint**: black --check + mypy
- **Test**: Python 3.9 / 3.11 / 3.12 matrix, coverage >= 60%
- **Build**: wheel + sdist + install verification

## Benchmarks

Run the benchmark suite:

```bash
python scripts/benchmark_suite.py           # quick mode
python scripts/benchmark_suite.py --full    # thorough (10x)
python scripts/benchmark_suite.py --markdown # output BENCHMARKS.md
```

Sample results (CPU, quick mode):

| Operation | Throughput |
|-----------|-----------|
| LFSR step | 2.25 Mstep/s |
| Bitstream encoder | 1.88 Mstep/s |
| LIF neuron step | 1.15 Mstep/s |
| vec_and (1024 words) | 45.67 Gbit/s |
| gpu_vec_mac (64x32x16w) | 6.15 GOP/s |

## Documentation

- [CHANGELOG.md](CHANGELOG.md) -- Version history
- [BENCHMARKS.md](BENCHMARKS.md) -- Performance benchmark results
- [docs/HARDWARE_GUIDE.md](docs/HARDWARE_GUIDE.md) -- FPGA deployment workflow
- [docs/API_REFERENCE.md](docs/API_REFERENCE.md) -- API documentation
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) -- Usage guide

## Install Extras

```bash
pip install -e ".[dev]"       # pytest, mypy, black
pip install -e ".[gpu]"       # CuPy CUDA acceleration
pip install -e ".[research]"  # networkx, onnx, torch
pip install -e ".[contrib]"   # speculative module deps
pip install -e ".[full]"      # networkx, onnx
```

## License

MIT
