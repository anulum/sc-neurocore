# NIR Integration Guide

SC-NeuroCore provides an [NIR](https://neuroir.org/) bridge for supported
graphs and a selected path toward FPGA synthesis. Import a supported graph,
simulate it, export compatible representations, and inspect emitted
SystemVerilog. Hardware deployment needs target-specific validation.

## What is NIR?

NIR (Neuromorphic Intermediate Representation) is an open standard for
exchanging spiking neural network models between frameworks. It defines
18 primitives (LIF, IF, CubaLIF, Affine, Conv, etc.) as a directed graph.
Libraries like Norse, snnTorch, and Lava-DL can export to NIR. SC-NeuroCore
can import supported graphs, run them, export compatible forms, and
prepare RTL for FPGA validation.

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

The table names the mapped primitives. Faithful exchange also depends on the
clock, reset convention, shapes and topology described below. A type mapping
alone does not establish equivalence with another framework's discrete solver.
With NIR 1.0.8, Conv2d declarations whose non-square kernel produces an
incorrect inferred output shape are refused with the conflicting geometry.

## Graph and execution round trips

`to_nir()` retains independently owned graph and primitive metadata and records
the timestep and reset convention as JSON text under `sc_neurocore.execution`
(schema `sc-neurocore.nir-execution.v1`). A default `from_nir(path)` restores
those settings. Explicit options that contradict the recorded settings fail.
A graph without this extension uses the SC reference defaults: `dt=1.0` and
reset to `v_reset`.

The `sc_neurocore.graph_order` extension (schema
`sc-neurocore.nir-order.v1`) retains node and edge order, including repeated
edges. This matters for summation and the choice of recurrent back edges:
HDF5 otherwise enumerates node groups alphabetically. Stale or malformed order
metadata is refused. Unchanged parameter arrays keep their values, shapes and
dtypes; changed parameters are exported from the current executable definition.
Convolution geometry is owned as integer scalars or tuples and exported in
that canonical form, including when the source supplied mutable NumPy arrays.
HDF5 normalises structural tuples to integer arrays. Tensor rank is restored
on graph edges, including singleton convolution and pooling dimensions.

The reference NIR reader/writer carries these metadata extensions. A different
consumer must implement their conventions; merely reading the metadata does
not make its solver agree. Time units are unchanged, with no automatic
millisecond/second conversion. Dynamic membrane/current/delay state and
optimizer state are not checkpoints in this format: an import starts fresh.

Real file tests cover primitive graphs, signed dense projections, pooling,
convolutions, nested graphs, branches and recurrence. Fractional delays are
rounded to steps and reported as approximations. Nested multi-port graphs used
as one node still consume/return their first port; `forward_multi()` is the
explicit named-port surface. These approximations are outside faithful
snnTorch sequence export.

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

The qualified profiles use Linux x86_64, Python 3.12.16 and CPU
`torch==2.14.1+cpu`. Their complete hash-pinned dependency sets live in
`requirements/ci-torch-cpu.txt` and `requirements/ci-nir-<profile>.txt`.
Install the profiles in separate environments:

| Profile | Packages | Boundary exercised |
| --- | --- | --- |
| `interop-snntorch` | `nir==1.0.8`, `nirtorch==2.6`, `snntorch==1.0.0` | Actual snnTorch → SC → NIR file → SC → snnTorch inference |
| `interop-sinabs` | `nir==1.0.4`, `nirtorch==2.0.2`, `sinabs==3.1.3` | Actual Sinabs export, SC reference execution and NIR file round trip; native Sinabs solver equivalence is not claimed |

Sinabs 3.1.3 requires NIR at most 1.0.4; NIRTorch 2.0.3 and later require
NIR at least 1.0.6. The separate legacy profile resolves this conflict.
It has no CubaLI primitive or reset-voltage field: nonzero reset voltages and
CubaLI export are refused where that schema cannot represent them.

### snnTorch and PyTorch inference

```python
import numpy as np
import snntorch as snn
import torch
from sc_neurocore.nir_bridge import from_snntorch, from_nir, to_nir, to_snntorch

model = torch.nn.Sequential(
    torch.nn.Linear(4, 6, dtype=torch.float64),
    snn.Synaptic(alpha=0.9, beta=0.8, init_hidden=True, reset_delay=False),
).double()
network = from_snntorch(model, torch.zeros(4, dtype=torch.float64), dt=1e-4)
to_nir(network, "network.nir")
restored = from_nir("network.nir")
foreign = to_snntorch(restored)

current = np.array([1.0, 2.0, 3.0, 4.0])
sc_output = restored.step({restored.input_nodes[0]: current})
with torch.no_grad():
    torch_output = foreign(torch.from_numpy(current))
```

The importer uses the public NIRTorch FX API with normalised cell parameters;
it does not depend on snnTorch's scalar-parameter NIR exporter succeeding.
It snapshots definitions without changing the original model's parameters,
training mode or live state. The returned network starts from reset state.

Supported source models are nonempty, flat `torch.nn.Sequential` sequences of
Linear, Leaky and Synaptic layers. Cells need `init_hidden=True`,
`output=False`, `reset_delay=False`, unit spike grading, and the standard ATan
or disabled surrogate spike decision. Input and model tensors must be finite
CPU float64 values, one vector per step. Forward hooks, custom spike decisions,
inhibition, state quantisation, mixed resets, batches and other layer types
are refused. Leaky permits beta in [0,1), or uniformly one for IF; Synaptic
requires alpha and beta in [0,1).

snnTorch's default scalar spike grading leaves binary outputs float32 even
for double membrane state. A multilayer double source sequence needs each
cell's `graded_spikes_factor = torch.ones(width, dtype=torch.float64)` before
the next double Linear layer. The importer checks that the actual source can
execute its declared input. Exported cells cast binary spikes exactly to
float64 between layers.

`to_snntorch()` returns independent real Torch Linear and snnTorch cells for a
single-input/single-output chain of Affine, Linear, IF, LIF and CubaLIF nodes.
It preserves the input gains, thresholds and supported decay factors. Leak
and reset potentials must be zero. Branches, merges, recurrence, complex
parameters, non-vector boundaries, unsupported decays and approximated imports
are refused. Membrane voltage keeps its units; a CubaLIF cell's internal
snnTorch synaptic state is scaled by `r*dt/tau_mem`. Reset exported cells with
their `reset_mem()` method. This is inference interchange, with no claim to
preserve optimizer state, surrogate gradients or a live simulation checkpoint.

Euler and decay-form arithmetic may round differently. The real multilayer
corpus requires identical spikes and membrane/current errors at most 1e-12
in float64. This is a measured corpus contract, not a universal guarantee at
arbitrarily close threshold boundaries. Float32 and GPU inference are outside
this qualified profile.

### Reset conventions

| Mode | Spike decision and reset |
| --- | --- |
| `reset` | Detect `v > threshold`, then set spiking voltages to `v_reset` |
| `subtract` | Detect `v > threshold`, then subtract one threshold |
| `snntorch-subtract` | Account for the previous membrane reset signal before detection, matching snnTorch's immediate subtract convention |

The last two differ when an earlier post-reset voltage still exceeds its
threshold. `from_snntorch()` records the third convention automatically;
ordinary SC subtract graphs are refused on snnTorch export. Hardware lowering
currently implements reset to `v_reset` and refuses both subtract variants.

### Stochastic-computing contracts

Floating Torch inference cannot preserve bitstream length, unipolar/bipolar
or random-source encoding, Q-format, rounding/overflow, random-source identity
or stream correlations. Passing a real SC-NIR document through the `scnir=`
argument is refused in both framework directions. Attached SC-NIR documents
and declared stream/precision semantics in graph or node metadata are also
refused. Keep these contracts in SC-NIR and use its validated hardware path;
do not interpret float inference as a stochastic hardware equivalent.

### Sinabs and other exporters

The legacy Sinabs tests call `sinabs.nir.to_nir(model, sample_data)` on a real
Linear → LIFSqueeze model and exercise its parameters, SC execution and HDF5
round trip. They also require a nontrivial native Sinabs/SC trace mismatch to
remain visible: Sinabs uses exponential decay and inclusive spike detection,
whereas this reference interpreter uses Euler integration and strict detection.
A successful import is not native Sinabs behavioral acceptance.

Norse, Rockpool, SpikingJelly and snnTorch recurrent layers are outside these
qualified framework profiles. A graph their exporters produce may be admitted
by `from_nir()`, but its solver, clock, reset, gains and recurrence conventions
must be checked separately. Hand-built framework-shaped fixtures do not qualify
a real framework exporter.

### Studio graph envelopes

The Studio JSON graph envelope is separate from a real NIR HDF5 file.
Version 3 preserves execution timestep, duration, graph seed, population drive,
layout, projection identity and seeded connectivity; versions 1, 2 and the
legacy `nir`/`0.1` envelope remain readable with their documented defaults.
Unknown versions and missing version-3 execution settings are refused.
`graph_to_envelope()` / `envelope_to_graph()` are tested through JSON and the
actual Studio simulation, including random connectivity and Poisson drive.
The real HDF5 surface remains `graph_to_nir_file()` / `nir_file_to_graph()`.

## Reproduce inference measurements

The committed `benchmarks/results/bench_nir_interop.json` recording
contains 30 measured traces after two warmups for actual snnTorch, SC reference,
SC restored from HDF5 NIR, and exported snnTorch. Both immediate reset conventions
must produce identical spikes before timing; final source/SC membrane error must
be at most 1e-12. Producer hashes bind the recording to the executed code.

From a checkout with the qualified snnTorch profile installed:

```bash
PYTHONPATH=src python benchmarks/bench_nir_interop.py \
  --iterations 30 --warmups 2 --steps 256 \
  --output benchmarks/results/bench_nir_interop.json
```

These are local CPU float64 inference regression measurements on a shared host.
They do not measure a stochastic bitstream, FPGA throughput or an isolated
production deployment. The separate NIR graph benchmark retains its identical
RTL comparison against the parent source snapshot. On 2026-10-30, host
`aaarthuus` (Intel Core i5-11600K), 30 cold-process samples per variant after
two warmups measured median public parsing and lowering at 266,281.5 ns for
parent `a21b41442` and 422,404.5 ns for this implementation: 58.63% overhead.
Retaining original definitions copies weight arrays beyond the executable
parameters, so copy work grows with weight size; metadata custody and per-graph
order/execution validation add work with metadata size and topology, but these
costs were not measured separately. Reusing retained-definition metadata
directly would alias two mutable containers; retaining their current independent
ownership requires another copy or a separately reviewed custody redesign.
Each sample lowers the graph 25 times and compiles it five times. These
shared-workstation medians describe local regression context rather than
production throughput.
The committed graph recording fingerprints the parser and node factories as
well as the graph modules.

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

The examples illustrate foreign graph conventions; they are not substitutes
for the qualified installed-framework profiles above.

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
