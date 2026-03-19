<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# SC-NeuroCore for Research Labs

A streamlined setup guide for neuroscience, hardware, and ML research
groups evaluating SC-NeuroCore for simulation, training, or FPGA deployment.

## 5-Minute Quick Start

```bash
pip install sc-neurocore
python -c "
from sc_neurocore import StochasticLIFNeuron
n = StochasticLIFNeuron()
spikes = sum(n.step(0.8) for _ in range(500))
print(f'{spikes} spikes in 500 steps')
"
```

That's it. NumPy is the only hard dependency. Everything else is optional.

## Which Install Path?

| Your goal | Install command | What you get |
|-----------|----------------|--------------|
| Explore neuron models | `pip install sc-neurocore` | 122 models, simulation, analysis |
| Train SNNs on GPU | `pip install sc-neurocore[research]` | + PyTorch training, matplotlib |
| Benchmark against Brian2 | `pip install sc-neurocore[accel]` | + Numba JIT (4x speedup at 1K neurons) |
| Deploy to FPGA | `pip install sc-neurocore` + [Yosys](https://github.com/YosysHQ/yosys) | + IR compiler, SystemVerilog emission |
| Import NIR models | `pip install sc-neurocore[nir]` | + [NIR](https://neuroir.org/) bridge (Norse, snnTorch, Lava-DL interop) |
| Full research stack | `pip install sc-neurocore[full]` | Everything above |

**Python 3.10 through 3.14** supported. Linux, macOS, Windows.

## NIR Interoperability

SC-NeuroCore is the first [NIR](https://neuroir.org/) backend targeting FPGA synthesis.
Import models from Norse, snnTorch, or Lava-DL via the NIR standard, then simulate
or compile to SystemVerilog. See the [NIR Integration Guide](nir_integration.md) and
the [NIR Bridge Notebook](https://github.com/anulum/sc-neurocore/blob/main/notebooks/05_nir_bridge.ipynb).

## For Computational Neuroscience Labs

SC-NeuroCore operates at a different abstraction than Brian2, NEST, or
NEURON. Where those tools solve differential equations for continuous
membrane voltage, SC-NeuroCore encodes values as stochastic bitstreams
and performs arithmetic with logic gates — directly mapping to FPGA
hardware.

**What you gain:** bit-true correspondence between your Python
simulation and synthesised Verilog. What fires in simulation fires
identically on silicon.

**What you keep:** familiar neuron models. SC-NeuroCore implements
122 models from McCulloch-Pitts (1943) through Hodgkin-Huxley,
Izhikevich, multi-compartment Hay L5 pyramidal, all the way to
hardware chip emulators (Loihi, TrueNorth, BrainScaleS, SpiNNaker,
Akida).

### First experiment: Brunel balanced network

```python
from sc_neurocore.model_zoo.configs import brunel_balanced_network

net = brunel_balanced_network()
net.run(duration_ms=1000, dt=0.1)
```

This creates an 800E/200I network with Izhikevich neurons, random
connectivity, and Poisson drive — the standard benchmark from
Brunel (2000). Compare firing rates against your Brian2 implementation.

### Spike train analysis

126 analysis functions, zero external dependencies:

```python
from sc_neurocore.analysis import (
    firing_rate, cv_isi, fano_factor,
    victor_purpura_distance, phase_locking_value,
    mutual_information, granger_causality,
    gpfa, spade_detect,
)
```

Covers the combined scope of Elephant and PySpike: statistics, distance
metrics, synchrony, information theory, causality, dimensionality
reduction, decoding, and pattern detection — all pure NumPy.

### Deeper reading

- [SC for Neuroscientists](SC_FOR_NEUROSCIENTISTS.md) — rate coding
  vs bitstream coding, model correspondence table
- [Tutorial: Brunel Network Translation](../tutorials/06_brunel_network_translation.md)
- [Tutorial: Spike Train Analysis](../tutorials/23_spike_train_analysis.md)

## For Hardware / FPGA Labs

SC-NeuroCore is the only open-source framework where the Python
simulation matches synthesisable Verilog RTL cycle-exactly.

### Pipeline

```
Train (PyTorch) → Quantise (Q8.8) → Simulate (bitstreams) → Compile (IR) → Synthesise (Yosys/Vivado)
```

### Quick synthesis check

```bash
pip install sc-neurocore
python examples/06_hdl_generation.py    # Generates Verilog
yosys -p "synth_xilinx" hdl/sc_neurocore_top.v   # Synthesis report
```

17 synthesisable Verilog modules. Yosys reports 7 382 LUTs for a
3-input 7-neuron core on Xilinx 7-series. MNIST 16→10 fits an
Artix-7 100T at ~56K LUTs.

### Formal verification

64 properties across 7 SymbiYosys formal modules (encoder, neuron,
synapse, dense layer, dotproduct, firing rate, AXI-Lite config).

### Prerequisites for physical deployment

- [Yosys](https://github.com/YosysHQ/yosys) (open source, synthesis + reports)
- Xilinx Vivado Design Suite (for Artix-7/Zynq bitstreams) or
  Lattice iCEcube2 (for iCE40)
- FPGA board (recommended: Arty A7-100T or PYNQ-Z2)

### Deeper reading

- [SC for Hardware Engineers](SC_FOR_HARDWARE_ENGINEERS.md) — gate
  reduction analysis, architecture walkthrough
- [Tutorial: FPGA in 20 Minutes](../tutorials/fpga_in_20_minutes.md)
- [Tutorial: Hardware Co-simulation](../tutorials/09_hardware_cosimulation.md)
- [FPGA Toolchain Guide](../hardware/FPGA_TOOLCHAIN_GUIDE.md)

## For ML / SNN Training Labs

SC-NeuroCore bridges float-domain training to hardware deployment.
Train with surrogate gradients in PyTorch, export to stochastic
bitstream weights, deploy on FPGA.

### Train a digit classifier

```bash
pip install sc-neurocore[research]
python examples/mnist_conv_train.py
```

Achieves 99.49% MNIST accuracy with a ConvSpikingNet using learnable
membrane time constants and threshold parameters.

### Export to hardware

```python
from sc_neurocore.training import to_sc_weights

sc_weights = to_sc_weights(trained_model)
# Weights are normalised to [0, 1] for bitstream encoding
```

No other SNN training library (snnTorch, Norse) provides this
train-to-hardware export path.

### Deeper reading

- [SC for ML Engineers](SC_FOR_ML_ENGINEERS.md) — positioning vs
  snnTorch/Norse, surrogate gradient details
- [Tutorial: Surrogate Gradient Training](../tutorials/03_surrogate_gradient_training.md)
- [Tutorial: MNIST SC Classification](../tutorials/07_mnist_sc_classification.md)

## Rust Engine (Optional, Recommended)

The Rust SIMD engine provides 512x real-time acceleration with
AVX-512, AVX2, or NEON auto-dispatch. 111 neuron models compiled
to native code with PyO3 bindings.

```bash
# From a source checkout
pip install -r requirements/maturin.txt
cd bridge
maturin develop --release
```

The engine is optional — all functionality works with the pure-Python
package. The engine adds speed, not features.

## Interactive Notebooks

| Notebook | What it shows |
|----------|---------------|
| [Quickstart (Colab)](https://colab.research.google.com/github/anulum/sc-neurocore/blob/main/notebooks/quickstart_colab.ipynb) | LIF neuron, dense layer, spike raster, SC convolution |
| [Neuron Explorer](https://github.com/anulum/sc-neurocore/blob/main/notebooks/04_neuron_explorer.ipynb) | Browse all 117 models, voltage traces, phase portraits, F-I curves |
| [HDC Symbolic Query](https://github.com/anulum/sc-neurocore/blob/main/notebooks/01_hdc_symbolic_query.ipynb) | Hyper-dimensional computing with 10K-bit vectors |
| [End-to-End Pipeline](https://github.com/anulum/sc-neurocore/blob/main/notebooks/03_end_to_end_pipeline.ipynb) | Encode → simulate → decode → visualise |
| [Fault-Tolerant Logic](https://github.com/anulum/sc-neurocore/blob/main/notebooks/02_fault_tolerant_logic.ipynb) | SC error resilience under bit-flip noise |

## Lab Setup Checklist

- [ ] `pip install sc-neurocore` — verify: `python -c "import sc_neurocore; print(sc_neurocore.__version__)"`
- [ ] Run quickstart: `python examples/01_basic_sc_encoding.py`
- [ ] Run your domain tutorial (neuroscience / hardware / ML — links above)
- [ ] Optional: build the Rust bridge from the repo checkout for acceleration
- [ ] Optional: install Yosys for FPGA synthesis reports
- [ ] Explore the [neuron model explorer notebook](https://github.com/anulum/sc-neurocore/blob/main/notebooks/04_neuron_explorer.ipynb)

## Licensing

SC-NeuroCore is **free for research and education** under AGPL-3.0.
Academic labs can use the full framework — 122 neuron models, Rust
engine, Verilog RTL, quantum modules, training pipeline — at no cost.

Proprietary integration requires a [commercial license](../pricing.md).
Academic discounts available — contact
[protoscience@anulum.li](mailto:protoscience@anulum.li).

## Support

- [Documentation](https://anulum.github.io/sc-neurocore/)
- [GitHub Discussions](https://github.com/anulum/sc-neurocore/discussions) — questions, ideas
- [GitHub Issues](https://github.com/anulum/sc-neurocore/issues) — bug reports
- Email: [protoscience@anulum.li](mailto:protoscience@anulum.li)
- ORCID: [0009-0009-3560-0851](https://orcid.org/0009-0009-3560-0851)
