# NIR Integration Guide

SC-NeuroCore is an [NIR](https://neuroir.org/) backend targeting FPGA synthesis.
Import any NIR graph, simulate it with SC-NeuroCore's stochastic computing
engine, export back to NIR, and emit SystemVerilog for hardware deployment.

## What is NIR?

NIR (Neuromorphic Intermediate Representation) is an open standard for
exchanging spiking neural network models between frameworks. It defines
18 primitives (LIF, IF, CubaLIF, Affine, Conv, etc.) as a directed graph.
Libraries like Norse, snnTorch, and Lava-DL can export to NIR. SC-NeuroCore
can import those graphs, run them, export back, and deploy them to FPGAs.

## Installation

```bash
pip install sc-neurocore[nir]
```

Or install separately:

```bash
pip install sc-neurocore nir
```

## Quick Start

```python
import numpy as np
import nir
from sc_neurocore.nir_bridge import from_nir, to_nir

# Load a NIR graph from file
network = from_nir("model.nir", dt=1.0)

# Or from a NIR graph object
graph = nir.read("model.nir")
network = from_nir(graph, dt=1.0)

# Run for 100 timesteps
results = network.run({"input": np.array([1.0, 0.5, 0.2])}, steps=100)

# Inspect output spikes
for step_output in results["output"]:
    print(step_output)

# Export back to NIR
graph_out = to_nir(network)
nir.write("exported.nir", graph_out)
```

## Supported NIR Primitives

| NIR Primitive | SC-NeuroCore Mapping | Notes |
|---|---|---|
| `Input` | `SCInputNode` | Graph entry point (passthrough) |
| `Output` | `SCOutputNode` | Graph exit point (collector) |
| `LIF` | `SCLIFNode` | Euler: v += ((v_leak - v) + R*I) * dt/tau |
| `IF` | `SCIFNode` | Euler: v += R*I*dt, fire when v > threshold |
| `LI` | `SCLINode` | Leaky integrator (no threshold) |
| `I` | `SCIntegratorNode` | Pure integrator: v += R*I*dt |
| `Affine` | `SCAffineNode` | W @ x + b (dense layer with bias) |
| `Linear` | `SCLinearNode` | W @ x (dense layer without bias) |
| `Scale` | `SCScaleNode` | Element-wise scaling |
| `Threshold` | `SCThresholdNode` | Spike generation |
| `Flatten` | `SCFlattenNode` | Tensor reshape with dim range |
| `NIRGraph` | `SCSubgraphNode` / `SCMultiPortSubgraphNode` | Nested subgraph (recursive, single or multi-port) |
| `CubaLIF` | `SCCubaLIFNode` | Current-based LIF with synaptic filter (dual tau) |
| `CubaLI` | `SCCubaLINode` | Current-based leaky integrator |
| `Delay` | `SCDelayNode` | Circular buffer delay (zero-delay passthrough supported) |
| `Conv1d` | `SCConv1dNode` | 1D convolution with stride, padding, dilation, groups |
| `Conv2d` | `SCConv2dNode` | 2D convolution with full parameter support |
| `SumPool2d` | `SCSumPool2dNode` | Spatial sum pooling over kernel windows |
| `AvgPool2d` | `SCAvgPool2dNode` | Average pooling (SumPool / kernel_area) |

All 18 NIR primitives are supported. 100% coverage of the NIR standard.
Bidirectional: `from_nir()` imports, `to_nir()` exports — full roundtrip
with parameter fidelity verified for all node types.

## Recurrent Connections

Graphs with cycles (feedback/recurrent connections) are automatically
handled. Back edges are detected by DFS and replaced with unit-delay
nodes that buffer the previous timestep's output. On export, the
original recurrent edges are reconstructed.

```python
# Example: LIF with recurrent feedback
edges = [
    ("input", "affine"),
    ("affine", "lif"),
    ("lif", "rec_weight"),  # forward
    ("rec_weight", "lif"),  # back edge (cycle)
    ("lif", "output"),
]
# from_nir() handles this automatically
network = from_nir(graph, dt=1.0)
# to_nir() reconstructs the original edges
graph_out = to_nir(network)
```

## Nested Subgraphs

NIR supports nested `NIRGraph` nodes. SC-NeuroCore wraps these as:
- `SCSubgraphNode` for single-input single-output subgraphs
- `SCMultiPortSubgraphNode` for multi-input/multi-output subgraphs

Both are handled automatically during import and export recursively.

## Building a NIR Graph Manually

```python
import numpy as np
import nir

nodes = {
    "input": nir.Input(input_type={"input": np.array([3])}),
    "affine": nir.Affine(
        weight=np.random.randn(2, 3).astype(np.float32),
        bias=np.zeros(2, dtype=np.float32),
    ),
    "lif": nir.LIF(
        tau=np.full(2, 20.0),
        r=np.ones(2),
        v_leak=np.zeros(2),
        v_threshold=np.ones(2),
    ),
    "output": nir.Output(output_type={"output": np.array([2])}),
}
edges = [("input", "affine"), ("affine", "lif"), ("lif", "output")]
graph = nir.NIRGraph(nodes=nodes, edges=edges)

# Save to file
nir.write("my_model.nir", graph)
```

Note: `input_type` and `output_type` use shape arrays (`np.array([n])`)
not data arrays.

## SCNetwork API

`from_nir()` returns an `SCNetwork` with these methods:

| Method | Description |
|--------|-------------|
| `step(inputs)` | Run one timestep. Returns dict of output arrays. |
| `run(inputs, steps=100)` | Run multiple timesteps. Returns dict of lists. |
| `reset()` | Reset all stateful nodes to initial conditions. |
| `summary()` | Print human-readable network topology. |
| `to_hardware(...)` | Lower the parsed network through the production FPGA compiler and return generated RTL plus SC-NIR metadata. |
| `topo_order` | Topologically sorted node execution order. |

For a one-object import-to-hardware workflow, use the classmethod facade:

```python
from sc_neurocore.nir_bridge import SCNNetwork

network = SCNNetwork.from_nir(graph, dt=1.0)
result = network.to_hardware(
    module_name="my_snn",
    data_width=16,
    fraction=8,
    bitstream_length=1024,
    target="artix7",
)

verilog = result.top_module
```

`SCNNetwork` is a public alias for the NIR bridge `SCNetwork`. The hardware
method delegates to `from_scnetwork()` and `compile_network_to_fpga()`, so it
preserves the same validation, quantisation, generated Verilog, and SC-NIR
metadata as the lower-level compiler path.

### Fan-in

When multiple edges converge on a single node, their outputs are summed
before being passed as input. This matches standard neural network
semantics for additive synaptic currents.

## Interoperability

### Import from Norse

```python
import torch
import norse.torch as norse

# Build a Norse SNN
model = norse.SequentialState(
    norse.LIFBoxCell(),
    norse.LILinearCell(128, 10),
)

# Export to NIR (requires sample_data for tracing)
graph = norse.to_nir(model, torch.randn(1, 128))

# Import into SC-NeuroCore
from sc_neurocore.nir_bridge import from_nir
network = from_nir(graph, dt=1.0)
```

**Note on Norse tau values:** Norse `export_nir.py` computes
`tau = dt / tau_inv` (default dt=0.001), which bakes the simulation
timestep into the exported time constants. Norse `import_nir.py` inverts
as `tau_inv = 1 / tau` without compensating for dt. We observed that
Norse's own export-import cycle produces different spike patterns on
identical input (verified with Norse 1.1.0). If importing Norse-exported
NIR graphs, verify that tau values match your expected dynamics. See
`examples/norse_nir_roundtrip.py` for a documented workaround.

### Import from snnTorch

```python
import torch
from snntorch.export_nir import export_to_nir

# Export snnTorch model to NIR
graph = export_to_nir(model, torch.randn(1, n_input))

# Import into SC-NeuroCore (dt must match snnTorch's hardcoded 1e-4)
from sc_neurocore.nir_bridge import from_nir
network = from_nir(graph, dt=1e-4, reset_mode="subtract")
```

**Note on snnTorch interop:** snnTorch uses subtract-reset (`v = v - threshold`)
but exports `v_reset=0` to NIR. Pass `reset_mode="subtract"` to match.
snnTorch hardcodes `dt=1e-4` in its export; use the same value in `from_nir()`.
The reproduced exporter profile uses `snntorch==0.9.4`, `nirtorch==2.6`, and
`nir==1.0.8`; snnTorch 1.0.0 currently fails inside its upstream NIR exporter
before SC-NeuroCore receives a graph.
Some configurations show spike mismatches (measured 6-8% across 600 steps on
12 configs) caused by float32 (torch) vs float64 (numpy) precision divergence
at threshold boundaries. The equations are algorithmically equivalent.

### Import from SpikingJelly

```python
import torch
from spikingjelly.activation_based import neuron, layer, functional
from spikingjelly.activation_based.nir_exchange import export_to_nir

# Build and export SpikingJelly model
functional.set_step_mode(model, "s")
graph = export_to_nir(model, torch.randn(1, n_input), dt=1e-4)

# Import into SC-NeuroCore (dt must match SpikingJelly's export dt)
from sc_neurocore.nir_bridge import from_nir
network = from_nir(graph, dt=1e-4)
```

**Note on SpikingJelly interop:** Verified exact spike match across 27
configurations (3 seeds, 3 tau values, 3 inputs, 1350 total steps, 0
mismatches). SpikingJelly exports LIFNode as `nir.LIF` with `tau=tau*dt`.
The reproduced profile uses upstream commit
`2797c5575515b59d7f09a3d5b732d6d25e148d13` with `nir==1.0.7`. NIR 1.0.8
removed the `input_type` and `output_type` constructor arguments that this
SpikingJelly exporter still supplies, so that latest/latest pairing currently
fails before SC-NeuroCore receives the graph. SpikingJelly's `CUBALIFNode` is
not yet mapped by their NIR export.

### Import from Sinabs

```python
import sinabs
import sinabs.nir

# Export sinabs model to NIR
graph = sinabs.nir.to_nir(model)

# Import into SC-NeuroCore (no dt baked into sinabs params)
from sc_neurocore.nir_bridge import from_nir
network = from_nir(graph, dt=1.0)
```

**Sinabs conventions:**
- Exports `nir.LIF` (for `LIF`/`LIFSqueeze`), `nir.IF` (for `IAF`/`IAFSqueeze`),
  `nir.LI` (for `ExpLeak`/`ExpLeakSqueeze`)
- Always exports `nir.Affine` (never `nir.Linear`) — fills bias with zeros
  even for bias-free layers
- Fixed: `r=1.0`, `v_leak=0.0` for all neuron types
- **No dt baked into parameters.** Tau is the physical time constant.
  Use `dt=1.0` for sinabs graphs.
- Does not export `CubaLIF`. No synaptic neuron types in sinabs NIR export.
- **Discretization difference:** Sinabs uses exponential decay
  (`alpha = exp(-dt/tau)`) while SC-NeuroCore uses Euler (`alpha = 1 - dt/tau`).
  With `dt=1.0, tau=10.0`: Euler gives 0.9, exact gives 0.9048 (~0.5% per step).
  After many steps the difference accumulates but remains bounded.

### Import from Rockpool

```python
from rockpool.nn.modules.torch.nir import to_nir

# Export rockpool model to NIR
graph = to_nir(model)

# Import into SC-NeuroCore (dt must match rockpool module's dt)
from sc_neurocore.nir_bridge import from_nir
network = from_nir(graph, dt=1e-3)  # rockpool tests use dt=1e-3
```

**Rockpool conventions:**
- Exports `nir.LIF` (for `LIFNeuronTorch`), `nir.CubaLIF` (for `LIFTorch`),
  `nir.LI` (for `ExpSynTorch`)
- **r encoding**: `r = tau * exp(-dt/tau) / dt` — encodes dt into the `r` field.
  You must use the same dt when calling `from_nir()`.
- `w_in` defaults to 1.0 (NIR default) for CubaLIF.
- **Weight transposition:** Rockpool transposes weights on NIR export
  (rockpool stores `(in, out)`, NIR expects `(out, in)`). Already transposed
  in the NIR graph, so `from_nir()` handles it transparently.
- `v_leak=0`, `v_reset=0` (NIR defaults) for all neuron types.
- **Discretization difference:** Rockpool uses exact exponential decay internally.
  With `dt=1e-3, tau=10.0`, the per-step Euler error is <1ppm, accumulating to
  <1% after 10,000 steps. At larger dt/tau ratios, divergence increases.

### Import from snnTorch RSynaptic (recurrent)

```python
import snntorch as snn
from snntorch.export_nir import export_to_nir

# RSynaptic model: recurrent CubaLIF
model = torch.nn.Sequential(
    torch.nn.Linear(4, 6),
    snn.RSynaptic(alpha=0.9, beta=0.8, all_to_all=True,
                  linear_features=6),
)
graph = export_to_nir(model, torch.randn(1, 4))

# Import — RSynaptic exports as NIRGraph subgraph
from sc_neurocore.nir_bridge import from_nir
network = from_nir(graph, dt=1e-4, reset_mode="subtract")
```

**snnTorch RSynaptic conventions:**
- Exports as a `nir.NIRGraph` subgraph: `Input → CubaLIF → Linear(w_rec) → CubaLIF → Output`
- SC-NeuroCore parses this as `SCSubgraphNode` with automatic cycle breaking
- Same dt=1e-4, subtract-reset, and float32/64 precision notes as snnTorch Synaptic above
- `RLeaky` exports similarly but with `nir.LIF` instead of `nir.CubaLIF`

### Lava-DL (Intel Loihi)

Lava-DL does not have a public NIR export function. It uses its own HDF5 network
exchange format. Interop with SC-NeuroCore requires manual graph construction or
conversion through an intermediate NIR graph. If Intel publishes NIR export support
in the future, `from_nir()` will handle it transparently.

## Framework dt/r Quick Reference

| Framework | `r` encoding | `dt` for `from_nir()` |
|---|---|---|
| Sinabs | `r = 1.0` (fixed) | `dt=1.0` |
| snnTorch | `r = tau_mem / dt` (dt=1e-4) | `dt=1e-4` |
| Rockpool | `r = tau * exp(-dt/tau) / dt` | Match module's dt |
| Norse | `r = 1.0` | `dt=1.0` |
| SpikingJelly | tau has dt baked in | Match export dt |

**The `r` field is the main cross-framework incompatibility.** The same physical
neuron produces different NIR `r` values depending on which framework exported it.
Always use the dt value that matches the exporting framework.

## `reset_mode` Parameter

`from_nir()` accepts `reset_mode` to control spike reset behavior:

| Mode | Behavior | Use when |
|---|---|---|
| `"reset"` (default) | `v = v_reset` | NIR spec default, Norse |
| `"subtract"` | `v = v - v_threshold` | snnTorch models |

## Runnable Demos

**CubaLIF + recurrent roundtrip** (synthetic graph, no extra deps):

```bash
pip install sc-neurocore nir
python examples/nir_roundtrip_demo.py
```

**Norse weights + CubaLIF + recurrent roundtrip** (requires Norse, torch):

```bash
pip install sc-neurocore nir norse torch
python examples/norse_nir_roundtrip.py
```

**SpikingJelly weights + LIF roundtrip** (requires SpikingJelly, torch):

```bash
pip install sc-neurocore 'nir==1.0.7' torch
pip install 'spikingjelly @ git+https://github.com/fangwei123456/spikingjelly.git@2797c5575515b59d7f09a3d5b732d6d25e148d13'
python examples/spikingjelly_nir_roundtrip.py
```

All demos verify full roundtrip: node names, edge sets, CubaLIF parameter
preservation, and file save/load. SC-NeuroCore's roundtrip preserves all
7 CubaLIF parameters exactly (bit-for-bit verified).

## FPGA Compilation

To compile an imported NIR graph directly to synthesisable Verilog RTL:

```python
from sc_neurocore.nir_bridge import (
    compile_network_to_fpga,
    from_nir,
    from_scnetwork,
)

network = from_nir("model.nir", dt=1.0)
neuron_graph = from_scnetwork(network, dt=1.0)
result = compile_network_to_fpga(neuron_graph, module_name="my_snn")

# result.top_module      → top-level interconnect Verilog
# result.neuron_modules  → per-type neuron Verilog modules
# result.weight_rom      → combined weight ROM Verilog
```

Or via CLI:

```bash
sc-neurocore compile-nir model.nir --target artix7 -o build/
```

See the [NIR/ONNX → FPGA Compilation Guide](nir_fpga_compilation.md) for
the full pipeline documentation, mathematical formalism, interconnect
selection, and quantisation details.
