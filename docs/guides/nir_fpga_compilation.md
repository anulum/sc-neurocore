<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->

# NIR/ONNX → FPGA Compilation Guide

Compile supported NIR spiking neural networks to synthesisable Verilog RTL with
a single command.  This guide covers the full pipeline: model import,
graph extraction, parameter quantisation, neuron module generation,
weight ROM artefact emission, and exact direct interconnect generation.

---

## 1. Mathematical Formalism

### 1.1 Canonical Neuron ODEs

The compiler maps five NIR neuron primitives to canonical ODE strings.
Each ODE is compiled to fixed-point Verilog via the equation compiler.

**LIF (Leaky Integrate-and-Fire):**

$$
\frac{dv}{dt} = -\frac{v - v_{\text{leak}}}{\tau} + \frac{I \cdot r}{\tau}
$$

Threshold condition: $v > v_{\text{threshold}}$.
Reset action: $v \leftarrow v_{\text{reset}}$.

**IF (Integrate-and-Fire):**

$$
\frac{dv}{dt} = I \cdot r
$$

No leakage term.  Fires when $v > v_{\text{threshold}}$, resets to $v_{\text{reset}}$.

**LI (Leaky Integrator):**

$$
\frac{dv}{dt} = -\frac{v - v_{\text{leak}}}{\tau} + \frac{I \cdot r}{\tau}
$$

Identical ODE to LIF but with no threshold/reset (non-spiking readout layer).

**CubaLIF (Current-Based LIF):**

$$
\frac{di_{\text{syn}}}{dt} = -\frac{i_{\text{syn}}}{\tau_{\text{syn}}} + I \cdot w_{\text{in}}
$$

$$
\frac{dv}{dt} = -\frac{v - v_{\text{leak}}}{\tau_{\text{mem}}} + \frac{i_{\text{syn}} \cdot r}{\tau_{\text{mem}}}
$$

Two state variables: synaptic current $i_{\text{syn}}$ and membrane
potential $v$.  The synaptic filter smooths the input with time constant
$\tau_{\text{syn}}$ before driving the membrane with time constant
$\tau_{\text{mem}}$.

**CubaLI (Current-Based Leaky Integrator):**

Same ODEs as CubaLIF but without threshold/reset.

### 1.2 Fixed-Point Quantisation

All parameters are encoded in Q-format fixed-point:

$$
x_{\text{int}} = \text{round}(x_{\text{float}} \times 2^{f})
$$

where $f$ is the number of fractional bits.  The representable range for
signed Q$m$.$f$ ($m = w - f - 1$ integer bits) is:

$$
\left[ -2^{m},\; 2^{m} - 2^{-f} \right]
$$

The minimum representable step (LSB resolution) is $2^{-f}$.

| Q-format | Width | Frac | Max Value | Min Step |
|----------|:-----:|:----:|:---------:|:--------:|
| Q8.8     |  16   |   8  |  127.996  | 0.00391  |
| Q4.12    |  16   |  12  |    7.9998 | 0.000244 |
| Q16.16   |  32   |  16  |  32767.99 | 1.53e-5  |
| Q4.4     |   8   |   4  |    7.9375 | 0.0625   |

Values outside the representable range are clamped with a warning.

### 1.3 Euler Discretisation

All ODEs are discretised with forward Euler:

$$
v[n+1] = v[n] + dt \cdot f(v[n], I[n])
$$

The timestep $dt$ must survive Q-format quantisation.  A critical warning
is emitted if $\text{round}(dt \times 2^{f}) = 0$, since this freezes
all dynamics.  For Q8.8, the minimum safe $dt$ is approximately $0.004$.

### 1.4 Interconnect Contract

The compiler emits explicit per-neuron direct wiring for all supported
networks.  This is resource-heavier than a routed event fabric, but it
preserves NIR affine semantics exactly.

For each destination neuron, the generated top module accumulates:

- external analogue inputs as `(input * weight) >>> fraction`;
- analogue source populations as `(v_out * weight) >>> fraction`;
- spiking source populations as `weight` on spike and zero otherwise;
- explicit bias terms when present.

All terms are summed in a widened signed accumulator and saturated back to
the target Q-format before entering the neuron module.

Weighted event-bus RTL is intentionally not emitted until address-event
fan-out, weight lookup, and destination accumulation have an audited
implementation.

---

## 2. Theoretical Context

### 2.1 Why This Pipeline Exists

Training SNNs in frameworks like snnTorch, Norse, Rockpool, or Sinabs
produces learned parameters (weights, time constants, thresholds) in
float32.  Deploying these to FPGA requires:

1. **Model import** — Parse the trained model into a graph IR.
2. **Parameter quantisation** — Convert float32 to fixed-point with
   overflow handling.
3. **RTL generation** — Emit synthesisable Verilog for each neuron type.
4. **Weight storage** — Encode all synaptic weights into a ROM module.
5. **Interconnect** — Wire neuron modules together.

Before this pipeline, users had to manually extract parameters, write
Verilog templates, and wire modules by hand.  The pipeline automates
all five steps.

### 2.2 NIR as the Universal Exchange Format

NIR (Neuromorphic Intermediate Representation) defines 18 primitives as
a directed graph.  All major SNN frameworks export to NIR.  By targeting
NIR, this pipeline supports any framework that can export a NIR graph.

The pipeline also accepts ONNX models with SNN custom ops via a
NIR-based shim: ONNX → NIR → NeuronGraph.

### 2.3 Relation to Existing Compiler

The `equation_compiler.compile_to_verilog()` function handles single
neurons.  The new pipeline extends this to entire networks by:

- Iterating over the graph topology
- Calling `compile_to_verilog()` once per unique neuron type
- Generating a weight ROM artefact for inspection and downstream flows
- Emitting a top-level direct interconnect module with per-neuron instances

---

## 3. Pipeline Position

### 3.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    User's Trained SNN                     │
│   (snnTorch / Norse / Rockpool / Sinabs / SpikingJelly)  │
└──────────────────────┬──────────────────────────────────┘
                       │ export_to_nir()
                       ▼
              ┌────────────────┐
              │   .nir file     │
              └───────┬────────┘
                      │ nir.read() + from_nir()
                      ▼
              ┌────────────────┐
              │  SCNetwork      │  ← parser.py
              └───────┬────────┘
                      │ from_scnetwork()
                      ▼
              ┌────────────────┐
              │  NeuronGraph    │  ← neuron_graph.py
              │  (populations   │
              │   + connections) │
              └───────┬────────┘
                      │ quantise_graph()
                      ▼
              ┌────────────────┐
              │ QuantisedGraph  │  ← quantise_params.py
              │ (Q-format ints) │
              └───────┬────────┘
                      │ compile_network_to_fpga()
                      ▼
        ┌─────────────┴─────────────┐
        │                           │
   ┌────┴────┐   ┌──────────┐  ┌───┴────────────┐
   │ Neuron   │   │ Weight   │  │ Top-Level      │
   │ Modules  │   │ ROM      │  │ Interconnect   │
   │ (*.v)    │   │ (.v)     │  │ (.v)           │
   └──────────┘   └──────────┘  └────────────────┘
```

### 3.2 Inputs

| Input | Type | Source |
|-------|------|--------|
| `.nir` file | NIR HDF5 graph | Any NIR-compatible framework |
| `dt` | float | Simulation timestep (must match export framework) |
| `data_width` | int | Fixed-point total bits (default: 16) |
| `fraction` | int | Fractional bits (default: 8) |
| `target` | str | FPGA target for hints (default: `"artix7"`) |

### 3.3 Outputs

| Output | Content |
|--------|---------|
| `<module_name>.v` | Top-level network interconnect |
| `sc_nir_<type>.v` | Per-type neuron Verilog (one per unique neuron type) |
| `sc_nir_weight_rom.v` | Combined weight ROM artefact for all connections |

---

## 4. Features

### 4.1 Supported Neuron Types

| NIR Primitive | Canonical Type | State Variables | Threshold |
|---------------|:--------------:|:---------------:|:---------:|
| `nir.LIF`     | `lif`          | `v`             | Yes       |
| `nir.IF`      | `if`           | `v`             | Yes       |
| `nir.LI`      | `li`           | `v`             | No        |
| `nir.CubaLIF` | `cuba_lif`     | `i_syn`, `v`    | Yes       |
| `nir.CubaLI`  | `cuba_li`      | `i_syn`, `v`    | No        |

### 4.2 Supported Connection Types

| NIR Primitive | Handling |
|---------------|----------|
| `nir.Affine`  | Weight matrix + bias → ConnectionSpec |
| `nir.Linear`  | Weight matrix (no bias) → ConnectionSpec |

### 4.3 Pass-Through Nodes

Input, Output, Scale, Flatten, Threshold, and Delay nodes are folded
into the graph metadata — they do not generate Verilog modules.

### 4.4 Quantisation Features

- Per-parameter overflow detection and clamping
- Per-parameter underflow detection
- dt quantisation-to-zero critical warning
- Configurable Q-format (Q4.4 through Q16.16)
- Signed two's complement encoding

### 4.5 Compilation Features

- One module per exact neuron type/parameter set (instantiated once per neuron)
- Combined weight ROM artefact with per-connection offset tracking
- Exact direct weighted interconnect with vector external input `I_ext_flat`
- Resource count reporting (total neurons, synapses)
- Warning accumulation across all pipeline stages

---

## 5. Usage Examples

### 5.1 Python API — Full Pipeline

```python
import numpy as np
import nir

from sc_neurocore.nir_bridge import (
    compile_network_to_fpga,
    from_nir,
    from_scnetwork,
)

# 1. Build or load a NIR graph
nodes = {
    "input": nir.Input(input_type={"input": np.array([4])}),
    "aff1": nir.Affine(
        weight=np.random.randn(8, 4).astype(np.float32),
        bias=np.zeros(8, dtype=np.float32),
    ),
    "lif1": nir.LIF(
        tau=np.full(8, 20.0),
        r=np.ones(8),
        v_leak=np.zeros(8),
        v_threshold=np.ones(8),
    ),
    "aff2": nir.Affine(
        weight=np.random.randn(2, 8).astype(np.float32),
        bias=np.zeros(2, dtype=np.float32),
    ),
    "lif2": nir.LIF(
        tau=np.full(2, 20.0),
        r=np.ones(2),
        v_leak=np.zeros(2),
        v_threshold=np.ones(2),
    ),
    "output": nir.Output(output_type={"output": np.array([2])}),
}
edges = [
    ("input", "aff1"), ("aff1", "lif1"),
    ("lif1", "aff2"), ("aff2", "lif2"),
    ("lif2", "output"),
]
graph = nir.NIRGraph(nodes=nodes, edges=edges)

# 2. Parse into SCNetwork
network = from_nir(graph, dt=1.0)

# 3. Extract NeuronGraph
neuron_graph = from_scnetwork(network, dt=1.0)
print(neuron_graph.summary())
# NeuronGraph: 2 populations, 2 connections
#   Total neurons:  10
#   Total synapses: 48
#   Neuron types:   lif

# 4. Compile to FPGA
result = compile_network_to_fpga(
    neuron_graph,
    module_name="my_snn",
    data_width=16,
    fraction=8,
    target="artix7",
)

# 5. Write output files
with open("my_snn.v", "w") as f:
    f.write(result.top_module)

for ntype, verilog in result.neuron_modules.items():
    with open(f"sc_nir_{ntype}.v", "w") as f:
        f.write(verilog)

with open("sc_nir_weight_rom.v", "w") as f:
    f.write(result.weight_rom)

print(f"Interconnect: {result.interconnect}")
print(f"Q-format: {result.q_format}")
print(f"Warnings: {result.warnings}")
```

### 5.2 CLI — One-Command Compilation

```bash
# Basic compilation (Q8.8, auto-interconnect)
sc-neurocore compile-nir model.nir -o build/

# High-precision Q16.16
sc-neurocore compile-nir model.nir --data-width 32 --fraction 16 -o build/

# Target-specific + custom module name
sc-neurocore compile-nir model.nir --target ice40 --module-name my_snn -o build/
```

Output:
```
[1/4] Loading model: model.nir
  Loaded 7 nodes
[2/4] Building NeuronGraph...
  10 neurons, 48 synapses
  Types: lif
[3/4] Compiling to Verilog (Q8.8)...
  Interconnect: direct
  Neuron modules: 1
[4/4] Output written to build/
  my_snn.v — top-level network
  sc_nir_lif.v — lif neuron module
  sc_nir_weight_rom.v — synaptic weight ROM
```

### 5.3 CubaLIF Network

```python
import nir, numpy as np
from sc_neurocore.nir_bridge import from_nir, from_scnetwork, compile_network_to_fpga

nodes = {
    "input": nir.Input(input_type={"input": np.array([3])}),
    "aff": nir.Affine(
        weight=np.random.randn(4, 3).astype(np.float32),
        bias=np.zeros(4, dtype=np.float32),
    ),
    "cuba": nir.CubaLIF(
        tau_syn=np.full(4, 5.0),
        tau_mem=np.full(4, 20.0),
        r=np.ones(4),
        v_leak=np.zeros(4),
        v_threshold=np.ones(4),
        w_in=np.ones(4),
    ),
    "output": nir.Output(output_type={"output": np.array([4])}),
}
edges = [("input", "aff"), ("aff", "cuba"), ("cuba", "output")]
graph = nir.NIRGraph(nodes=nodes, edges=edges)

net = from_nir(graph, dt=1.0)
ng = from_scnetwork(net, dt=1.0)
result = compile_network_to_fpga(ng, module_name="cuba_net")

# Generated: cuba_net.v, sc_nir_cuba_lif.v (with dual state variables),
#            sc_nir_weight_rom.v
```

### 5.4 Quantisation-Only (No Verilog)

```python
from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.nir_bridge import from_nir, from_scnetwork, quantise_graph

network = from_nir("model.nir", dt=1.0)
ng = from_scnetwork(network)
q = Q88(data_width=16, fraction=8)
qg = quantise_graph(ng, q)

print(f"Warnings: {qg.warnings}")
for pop in qg.populations:
    for pname, pval in pop.params.items():
        print(f"  {pop.name}.{pname}: min={pval.min()}, max={pval.max()}")
```

### 5.5 Mixed-Type Network

```python
import nir, numpy as np
from sc_neurocore.nir_bridge import from_nir, from_scnetwork, compile_network_to_fpga

nodes = {
    "input": nir.Input(input_type={"input": np.array([4])}),
    "aff1": nir.Affine(weight=np.random.randn(6, 4).astype(np.float32),
                       bias=np.zeros(6, dtype=np.float32)),
    "if_layer": nir.IF(r=np.ones(6), v_threshold=np.ones(6)),
    "aff2": nir.Affine(weight=np.random.randn(3, 6).astype(np.float32),
                       bias=np.zeros(3, dtype=np.float32)),
    "lif_layer": nir.LIF(tau=np.full(3, 15.0), r=np.ones(3),
                         v_leak=np.zeros(3), v_threshold=np.ones(3)),
    "output": nir.Output(output_type={"output": np.array([3])}),
}
edges = [("input", "aff1"), ("aff1", "if_layer"),
         ("if_layer", "aff2"), ("aff2", "lif_layer"),
         ("lif_layer", "output")]

result = compile_network_to_fpga(
    from_scnetwork(from_nir(nir.NIRGraph(nodes=nodes, edges=edges), dt=1.0)),
)

assert "if" in result.neuron_modules    # sc_nir_if.v
assert "lif" in result.neuron_modules   # sc_nir_lif.v
```

---

## 6. Technical Reference

### 6.1 `neuron_graph.py` — Data Structures

#### `NeuronSpec`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Unique population name (matches NIR node name) |
| `neuron_type` | `str` | `"lif"`, `"if"`, `"li"`, `"cuba_lif"`, `"cuba_li"` |
| `n_neurons` | `int` | Population size |
| `params` | `dict[str, np.ndarray]` | Canonical parameters |
| `dt` | `float` | Timestep |

#### `ConnectionSpec`

| Field | Type | Description |
|-------|------|-------------|
| `src` | `str` | Source population name |
| `dst` | `str` | Destination population name |
| `weights` | `np.ndarray` | Shape `(n_dst, n_src)`, float32 |
| `bias` | `np.ndarray \| None` | Shape `(n_dst,)`, optional |

#### `NeuronGraph`

| Field | Type | Description |
|-------|------|-------------|
| `populations` | `list[NeuronSpec]` | Ordered (topological) populations |
| `connections` | `list[ConnectionSpec]` | Weighted edges |
| `input_pop` | `str` | Input population name |
| `output_pop` | `str` | Output population name |
| `dt` | `float` | Global timestep |

Properties: `total_neurons`, `total_synapses`, `neuron_types`.

#### `from_scnetwork(network, dt=None) → NeuronGraph`

Walks the topologically-sorted node list and partitions nodes into
populations (neuron nodes) and connections (weight-carrying nodes).

### 6.2 `quantise_params.py` — Quantisation

#### `QuantisedGraph`

| Field | Type | Description |
|-------|------|-------------|
| `populations` | `list[NeuronSpec]` | Q-encoded params |
| `connections` | `list[ConnectionSpec]` | Q-encoded weights |
| `q` | `Q88` | Format configuration |
| `warnings` | `list[str]` | Overflow/underflow messages |

#### `quantise_graph(graph, q) → QuantisedGraph`

Encodes all float parameters to Q-format integers.  Clamps out-of-range
values and accumulates warnings.

### 6.3 `fpga_compiler.py` — Network Compiler

#### `NetworkCompilationResult`

| Field | Type | Description |
|-------|------|-------------|
| `neuron_modules` | `dict[str, str]` | Type → Verilog source |
| `weight_rom` | `str` | Combined weight ROM source |
| `top_module` | `str` | Top-level interconnect source |
| `module_name` | `str` | Top module name |
| `total_neurons` | `int` | Total neuron count |
| `total_synapses` | `int` | Total synapse count |
| `q_format` | `str` | e.g. `"Q8.8"` |
| `interconnect` | `str` | `"direct"` or `"aer"` |
| `warnings` | `list[str]` | Accumulated warnings |

#### `compile_network_to_fpga(graph, *, module_name, data_width, fraction, target) → NetworkCompilationResult`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `graph` | — | `NeuronGraph` input |
| `module_name` | `"sc_nir_network"` | Top Verilog module name |
| `data_width` | `16` | Fixed-point total width |
| `fraction` | `8` | Fractional bits |
| `target` | `"artix7"` | FPGA target hint |

### 6.4 CLI: `compile-nir`

```
sc-neurocore compile-nir <model> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `model` | — | `.nir` or `.onnx` file path |
| `--target` | `ice40` | FPGA target |
| `-o, --output` | `build` | Output directory |
| `--module-name` | `sc_equation_neuron` | Top module name |
| `--dt` | `1.0` | Simulation timestep |

---

## 7. Performance and Resource Notes

### 7.1 Compilation Time

The direct interconnect grows with the number of neurons and synapses.  Record
compile-time measurements from the target host before using this table in a
report:

| Network | Neurons | Synapses | Compile Time |
|---------|:-------:|:--------:|:------------:|
| 3-layer LIF (4→8→2) | 10 | 48 | measure locally |
| CubaLIF (3→4) | 4 | 12 | measure locally |
| Mixed IF+LIF (4→6→3) | 9 | 42 | measure locally |
| Large LIF (4→74→2) | 76 (direct) | 312 | measure locally |

### 7.2 Generated Verilog Size

| Network | Top Module | Neuron Module(s) | Weight ROM Artefact | Total |
|---------|:----------:|:-----------------:|:----------:|:-----:|
| 3-layer LIF | measure locally | generated | generated | measure locally |
| CubaLIF | measure locally | generated | generated | measure locally |
| Mixed IF+LIF | measure locally | generated | generated | measure locally |

### 7.3 Estimated FPGA Resources (Q8.8, Artix-7)

Per the `equation_compiler` resource estimator:

| Neuron Type | LUTs | FFs | DSPs | Notes |
|-------------|:----:|:---:|:----:|-------|
| LIF | ~120 | ~32 | 3 | 1 state var, 3 multiplies |
| IF | ~60 | ~16 | 1 | 1 state var, 1 multiply |
| CubaLIF | ~240 | ~64 | 6 | 2 state vars, 6 multiplies |

Weight ROM artefact size depends on network size:

| Synapses | ROM Entries | BRAM 18K | Notes |
|:--------:|:----------:|:--------:|-------|
| ≤1024 | ≤1024 | 1 | Single BRAM tile |
| ≤4096 | ≤4096 | 2 | Two 18Kb tiles |
| ≤16384 | ≤16384 | 4 | Switches to 36Kb tiles |

---

## 8. Citations

1. **NIR specification:**
   Pedersen, J. E. *et al.* "Neuromorphic Intermediate Representation:
   A Unified Instruction Set for Interoperable Brain-Inspired Computing."
   *arXiv:2311.14641*, 2023.

2. **LIF neuron model:**
   Lapicque, L. "Recherches quantitatives sur l'excitation électrique
   des nerfs traitée comme une polarisation." *J. Physiol. Pathol. Gén.*,
   9:620–635, 1907.

3. **CubaLIF (current-based synapse):**
   Rotter, S. and Diesmann, M. "Exact digital simulation of time-
   invariant linear systems with applications to neuronal modeling."
   *Biol. Cybern.*, 81:381–402, 1999.

4. **Fixed-point quantisation for SNNs:**
   Rueckauer, B. *et al.* "Conversion of continuous-valued deep networks
   to efficient event-driven networks for image classification."
   *Front. Neurosci.*, 11:682, 2017.

5. **snnTorch framework:**
   Eshraghian, J. K. *et al.* "Training spiking neural networks using
   lessons from deep learning." *Proc. IEEE*, 111(9):1016–1054, 2023.

7. **Norse framework:**
   Pehle, C. and Pedersen, J. E. "Norse — A deep learning library for
   spiking neural networks." *Zenodo*, DOI: 10.5281/zenodo.4422025, 2021.

8. **Euler discretisation stability for neuron ODEs:**
   Hansel, D. *et al.* "On numerical simulations of integrate-and-fire
   neural networks." *Neural Comput.*, 10(2):467–483, 1998.

---

## Cross-References

- [NIR Integration Guide](nir_integration.md) — Import/export, framework interop, reset modes
- [Deployment Guide](deployment_guide.md) — Resource estimation, constraints, bitstream flow
- [Compiler Intelligence Guide](compiler_intelligence.md) — Weight ROM, quantisation sweep, HLS export
- [Hardware Profiles Guide](hardware_profiles.md) — 65 target platform profiles
- [Static Analysis Guide](static_analysis_guide.md) — Guard bits, overflow proof
- [Precision Modes Guide](precision_modes.md) — 11 Q-format modes
- [Pipeline & Adaptive Precision Guide](pipeline_adaptive_precision.md) — Pipeline stages, dual-datapath LP/HP
