<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore — Supported workflows
-->

# Supported Workflows

Four things you can do with `pip install sc-neurocore` and nothing else. Each
block below is run by `tests/test_supported_workflows.py` from the source tree
and by `tests/test_studio_distribution.py` from a built wheel installed into a
separate environment, outside the repository; the test compares the printed
lines with the expected output shown here. If a block and its output disagree,
the test fails, not the reader.

What "supported" means here is narrow: this input gives this output on the
interpreters the test matrix runs (Python 3.10 to 3.14, Linux). It is not a
statement about speed, about other inputs, or about hardware.

## 1. Simulate a stochastic neuron

Install profile: base package.

<!-- workflow: stochastic-neuron -->
```python
from sc_neurocore import StochasticLIFNeuron

neuron = StochasticLIFNeuron(v_threshold=1.0, tau_mem=20.0, noise_std=0.0)
spikes = sum(neuron.step(0.8) for _ in range(500))
print(f"{spikes} spikes in 500 steps")
```

Expected output:

```text
250 spikes in 500 steps
```

Boundary: with `noise_std=0.0` the run is deterministic. With noise, pass a
seed to reproduce a run.

## 2. Multiply two probabilities with bitstreams

Install profile: base package.

<!-- workflow: bitstream-product -->
```python
from sc_neurocore import RNG, bitstream_to_probability, generate_bernoulli_bitstream

a = generate_bernoulli_bitstream(0.25, 65536, rng=RNG(1))
b = generate_bernoulli_bitstream(0.5, 65536, rng=RNG(2))
print(f"a = {bitstream_to_probability(a):.3f}")
print(f"b = {bitstream_to_probability(b):.3f}")
print(f"a AND b = {bitstream_to_probability(a & b):.3f}  (exact product 0.125)")
print(f"a AND a = {bitstream_to_probability(a & a):.3f}  (a stream is not independent of itself)")
```

Expected output:

```text
a = 0.249
b = 0.501
a AND b = 0.126  (exact product 0.125)
a AND a = 0.249  (a stream is not independent of itself)
```

Boundary: an AND gate multiplies only streams that are independent. The last
line shows what a shared stream gives instead. The error of the product
shrinks with the stream length; 65,536 bits give about three decimal places.

## 3. Define a neuron by its equation

Install profile: base package.

<!-- workflow: equation-neuron -->
```python
from sc_neurocore.neurons.equation_builder import from_equations

neuron = from_equations(
    "dv/dt = (-(v - E_L) + R * I) / tau",
    threshold="v > -50",
    reset="v = -65",
    params={"E_L": -65.0, "R": 10.0, "tau": 10.0},
    init={"v": -65.0},
    dt=0.1,
)
spikes = sum(neuron.step(I=2.0) for _ in range(2000))
print(f"{spikes} spikes in 2000 steps, v = {neuron.state['v']:.4f}")
try:
    from_equations("dv/dt = v.tofile(w)")
except ValueError as refusal:
    print("refused:", str(refusal).split(":")[0])
```

Expected output:

```text
14 spikes in 2000 steps, v = -55.0977
refused: Only a function under a plain name can be called in an equation
```

Boundary: an equation is arithmetic over names, the functions of the equation
namespace, indexing, and the attributes `real` and `imag`. Anything else is
refused when the neuron is built. The default integrator is forward Euler.

## 4. Lower that neuron to Verilog

Install profile: base package. No simulator or synthesis tool is needed to
generate the text; checking it against the Python model needs Icarus Verilog
and is a source-checkout workflow.

<!-- workflow: equation-to-verilog -->
```python
from sc_neurocore.compiler import compile_to_verilog
from sc_neurocore.neurons.equation_builder import from_equations

neuron = from_equations(
    "dv/dt = (-(v - E_L) + R * I) / tau",
    threshold="v > -50",
    reset="v = -65",
    params={"E_L": -65.0, "R": 10.0, "tau": 10.0},
    init={"v": -65.0},
    dt=0.1,
)
rtl = compile_to_verilog(neuron, module_name="lif_from_equation", data_width=32, fraction=16)
lines = rtl.splitlines()
print(next(line for line in lines if line.startswith("module ")).split("(")[0].strip())
print(f"{len(lines)} lines, ends with {lines[-1].strip()!r}")
```

Expected output:

```text
module lif_from_equation #
63 lines, ends with 'endmodule'
```

Boundary: this is generated fixed-point RTL text (Q16.16 here). It is not a
synthesis result, a timing result or a device measurement; those have their own
committed reports and are stated per model in the
[per-model evidence table](../api/model_fidelity_status.md).

## What each way of installing gives you

| You install | You get |
| --- | --- |
| `pip install sc-neurocore` | The four workflows above and the rest of the core package: neurons, synapses, layers, sources, recorders, the equation builder and the compilers. NumPy and SciPy provide numerical operations; the complete base dependencies and Python version conditions are declared in [`pyproject.toml`](https://github.com/anulum/sc-neurocore/blob/main/pyproject.toml). |
| An extra, for example `pip install "sc-neurocore[studio]"` | One optional area and its dependencies. The [optional dependency matrix](optional_dependency_matrix.md) lists each extra, the import it enables and the job that tests it. |
| The engine wheel (`sc_neurocore_engine`) | The Rust engine. The Python package runs without it. |
| A source checkout | Everything in the wheel plus the research subpackages listed below, the Julia, Go and Mojo lanes, the benchmarks and the hardware flows. |

These subpackages are in the repository and are not in the wheel:

<!-- source-only-subpackages -->
`audio`, `bio`, `chaos`, `dashboard`, `drivers`, `experiments`, `generative`,
`interfaces`, `optics`, `physics`, `robotics`, `sleep`, `swarm`, `viz`,
`analysis`, `world_model`.
