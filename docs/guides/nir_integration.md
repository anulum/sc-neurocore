# NIR Integration Guide

SC-NeuroCore is the first [NIR](https://neuroir.org/) backend that targets
FPGA synthesis. Import any NIR graph, simulate it with SC-NeuroCore's
stochastic computing engine, export back to NIR, and emit SystemVerilog
for hardware deployment.

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
| `topo_order` | Topologically sorted node execution order. |

### Fan-in

When multiple edges converge on a single node, their outputs are summed
before being passed as input. This matches standard neural network
semantics for additive synaptic currents.

## Interoperability

### Import from Norse

```python
import norse.torch as norse
import nir

# Train a Norse SNN
model = norse.SequentialState(
    norse.LIFBoxCell(),
    norse.LILinearCell(128, 10),
)

# Export to NIR (Norse >= 1.0)
graph = norse.to_nir(model)

# Import into SC-NeuroCore
from sc_neurocore.nir_bridge import from_nir
network = from_nir(graph, dt=1.0)
```

### Import from snnTorch

```python
import snntorch as snn
import nir

# Export snnTorch model to NIR
graph = snn.export_to_nir(model, sample_data)

# Import into SC-NeuroCore
from sc_neurocore.nir_bridge import from_nir
network = from_nir(graph, dt=1.0)
```

## Runnable Demo

See `examples/nir_roundtrip_demo.py` for a complete CubaLIF + recurrent
connection roundtrip with parameter verification.

```bash
pip install sc-neurocore nir
python examples/nir_roundtrip_demo.py
```
