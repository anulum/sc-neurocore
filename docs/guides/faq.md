# Frequently Asked Questions

## Installation

### Why is the first import slow?

SC-NeuroCore lazy-loads neuron model modules on first access. The
initial import takes ~10s (down from 200s in v3.12). Subsequent imports
within the same process are instant (<1ms).

### Do I need the Rust engine?

No. The Rust engine (`sc-neurocore-engine`) is optional. All Python
functionality works without it. The committed Brunel balanced-network
benchmark artifact `benchmarks/results/rust_scaling_benchmark.json` records
39-202x speedups against Brian2 for its 10K-100K rows; treat that as
workload-specific evidence, not a universal speed claim for every operation.

Use `sc-neurocore info` to see whether the current environment can import
`sc_neurocore_engine`. For source checkouts, build the local bridge with
`maturin develop --release`; the base `pip install sc-neurocore` path does not
require the engine.

### Do I need a GPU?

No. SC-NeuroCore runs on CPU by default. GPU acceleration is available
via CuPy (`pip install sc-neurocore[gpu]`) or PyTorch for surrogate
gradient training (`pip install sc-neurocore[training]`).

### Which Python versions are supported?

Python 3.10–3.14. CI tests all five versions on every commit.

---

## Usage

### How do I define a custom neuron model?

Use `EquationNeuron` with ODE strings (Brian2-style):

```python
from sc_neurocore.neurons.equation_builder import from_equations

neuron = from_equations(
    "dv/dt = -(v - E_L)/tau_m + I/C",
    threshold="v > -50",
    reset="v = -65",
    params=dict(E_L=-65.0, tau_m=10.0, C=1.0),
    init=dict(v=-65.0),
)
```

See [Notebook 08](https://github.com/anulum/sc-neurocore/blob/main/notebooks/08_equation_to_verilog.ipynb).

### How do I compile a neuron to FPGA?

```python
from sc_neurocore.compiler.equation_compiler import compile_to_verilog

verilog = compile_to_verilog(neuron, module_name="my_lif")
```

The output is synthesisable Q8.8 signed fixed-point Verilog. Target
ice40, ECP5, Gowin, or Xilinx via `sc-neurocore deploy model.nir --target ice40`.

### What does 100% test coverage mean?

100% line coverage on the core package (`src/sc_neurocore/`), excluding
hardware-dependent modules (GPU, Lava, PYNQ), optional dependencies
(PyTorch, JAX, NIR), neuron model files (tested via model-agnostic
step/reset), and the Studio web UI. The omitted modules are listed in
`pyproject.toml [tool.coverage.run] omit`.

### What is the FIM feedback parameter?

`Network(fim_lambda=λ)` enables Fisher Information Metric self-observation
feedback. After each plasticity step, each neuron's outgoing weights are
corrected by `ΔW -= λ · (activity_i - μ) / N`, pulling activity toward
the population mean. Derived from validated quantum control experiments
showing FIM alone synchronises networks (K=0, λ≥8).

Default is 0 (disabled). Recommended operating range: λ ≈ 2–3 × λ_c
where λ_c ≈ 0.149·N.

---

## Stochastic Computing

### Why use stochastic computing?

1. **Multiplication is one AND gate** (vs ~100 LUTs for fixed-point)
2. **Fault tolerance**: a stuck bit shifts probability by 1/L, not 50%
3. **Noise tolerance**: inherent to the representation
4. **Area efficiency**: thousands of neurons per FPGA

Trade-off: precision scales as O(1/√L) with bitstream length L. For 1%
error, L ≈ 2,500 (Bernoulli) or L ≈ 100 (Sobol quasi-random).

### Unipolar vs bipolar — when to use which?

- **Unipolar** [0,1]: AND gate = multiply. Use for weights ≥ 0.
- **Bipolar** [-1,1]: XNOR gate = multiply. Use for signed weights.

See [Notebook 14](https://github.com/anulum/sc-neurocore/blob/main/notebooks/14_sc_arithmetic_theory.ipynb).

---

## FPGA

### What FPGA boards are supported?

SC-NeuroCore generates Verilog RTL, not board-specific bitstreams.
Synthesis targets:

| Target | Toolchain | Tested |
|--------|-----------|--------|
| ice40 | Yosys + nextpnr | Yes (CI) |
| ECP5 | Yosys + nextpnr | Yes |
| Gowin | Yosys | Partial |
| Artix-7 | Vivado | Yes |
| Zynq | Vivado | Project TCL |

### How many neurons fit on an FPGA?

Rough estimates for a single LIF neuron at Q8.8:

| FPGA | LUTs available | LIF neurons (est.) |
|------|---------------|-------------------|
| iCE40 HX8K | 7,680 | ~150 |
| ECP5 85K | 84,000 | ~1,600 |
| Artix-7 100T | 63,400 | ~1,200 |
| Zynq 7020 | 53,200 | ~1,000 |

These are conservative estimates. Actual density depends on connectivity,
bitstream length, and whether DSP48 blocks are used for multiplies.

---

## Troubleshooting

### `ImportError: cannot import name 'sc_neurocore_engine'`

The Rust engine is not installed. Either:
- Build the local source bridge with `maturin develop --release` from the
  checkout.
- Install a matching release wheel when one is provided for your platform.
- Or use Python backend: `net.run(backend="python")`

### `SCEncodingError: Probability p must be in [0,1]`

Input to `BitstreamEncoder.encode()` or `generate_bernoulli_bitstream()`
is outside [0,1]. Check weight normalisation.

### Tests fail with `ModuleNotFoundError: torch`

PyTorch-dependent tests are skipped automatically when torch is not
installed. If you need training features: `pip install sc-neurocore[training]`

### `CuPy not available` warning

GPU acceleration requires CuPy: `pip install sc-neurocore[gpu]`.
The warning is informational; CPU computation continues normally.
