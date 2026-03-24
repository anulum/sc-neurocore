<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 30: NIR Bridge — Cross-Framework SNN Interoperability

SC-NeuroCore implements the [NIR](https://neuroir.org/) (Neuromorphic Intermediate
Representation) standard with 100% primitive coverage (18/18) and verified interop
with Norse, snnTorch, SpikingJelly, Sinabs, and Rockpool. This tutorial walks through
importing, running, exporting, and deploying NIR graphs.

## What is NIR?

NIR defines a common graph format for spiking neural networks. A model trained in
snnTorch can be exported to NIR, imported into SC-NeuroCore, simulated with stochastic
bitstreams, and compiled to FPGA via the IR compiler. NIR is the bridge between
GPU-trained models and hardware deployment.

## Prerequisites

```bash
pip install sc-neurocore nir
```

## 1. Import a Simple NIR Graph

```python
import numpy as np
import nir
from sc_neurocore.nir_bridge import from_nir, to_nir

# Build a 3-input LIF network manually
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

# Import into SC-NeuroCore
network = from_nir(graph, dt=1.0)
network.summary()
```

## 2. Run the Network

```python
# Step-by-step execution
for t in range(100):
    out = network.step({"input": np.array([1.0, 0.5, 0.2])})
    if np.any(out["output"] > 0):
        print(f"  t={t}: spike! output={out['output']}")

# Or batch execution
network.reset()
results = network.run({"input": np.array([1.0, 0.5, 0.2])}, steps=200)
total_spikes = sum(np.sum(s) for s in results["output"])
print(f"Total spikes in 200 steps: {total_spikes}")
```

## 3. Export Back to NIR

SC-NeuroCore's roundtrip preserves all parameters bit-for-bit:

```python
graph_out = to_nir(network)

# Verify roundtrip fidelity
orig_lif = graph.nodes["lif"]
exported_lif = graph_out.nodes["lif"]
assert np.allclose(orig_lif.tau, exported_lif.tau)
assert np.allclose(orig_lif.r, exported_lif.r)
assert np.allclose(orig_lif.v_threshold, exported_lif.v_threshold)
print("Roundtrip: all LIF parameters match exactly")

# Save to file
nir.write("my_model.nir", graph_out)
```

## 4. CubaLIF with Recurrent Connections

Many real SNN models use current-based LIF neurons with feedback. NIR represents
recurrent edges as graph cycles. SC-NeuroCore automatically detects and handles them.

```python
nodes = {
    "input": nir.Input(input_type={"input": np.array([4])}),
    "affine": nir.Affine(
        weight=np.random.randn(3, 4).astype(np.float32),
        bias=np.zeros(3, dtype=np.float32),
    ),
    "cuba": nir.CubaLIF(
        tau_mem=np.full(3, 20.0),
        tau_syn=np.full(3, 10.0),
        r=np.ones(3),
        v_leak=np.zeros(3),
        v_threshold=np.ones(3),
        w_in=np.ones(3),
    ),
    "rec": nir.Linear(weight=np.eye(3, dtype=np.float32) * 0.5),
    "output": nir.Output(output_type={"output": np.array([3])}),
}
edges = [
    ("input", "affine"),
    ("affine", "cuba"),
    ("cuba", "rec"),       # forward to recurrent weights
    ("rec", "cuba"),       # back edge (cycle) — automatic delay insertion
    ("cuba", "output"),
]
graph = nir.NIRGraph(nodes=nodes, edges=edges)
network = from_nir(graph, dt=1.0)

# The network detects the cycle and inserts a unit delay
results = network.run({"input": np.ones(4) * 2.0}, steps=100)
spikes = sum(np.sum(s) for s in results["output"])
print(f"CubaLIF + recurrent: {spikes} spikes in 100 steps")
```

## 5. Import from snnTorch

snnTorch models export to NIR with `dt=1e-4` hardcoded and subtract-reset:

```python
import torch
from snntorch.export_nir import export_to_nir

# Assume `model` is a trained snnTorch Sequential model
graph = export_to_nir(model, torch.randn(1, n_input))

# dt must match snnTorch's 1e-4, reset_mode must be "subtract"
network = from_nir(graph, dt=1e-4, reset_mode="subtract")
```

!!! warning "Float precision"
    snnTorch uses float32 (PyTorch), SC-NeuroCore uses float64 (NumPy).
    Expect 6-8% spike mismatches at threshold boundaries over long runs.
    The equations are algorithmically equivalent.

## 6. Import from Norse

```python
import torch
import norse.torch as norse

model = norse.SequentialState(
    norse.LIFBoxCell(),
    norse.LILinearCell(128, 10),
)
graph = norse.to_nir(model, torch.randn(1, 128))
network = from_nir(graph, dt=1.0)
```

!!! note "Norse tau observation"
    Norse `export_nir.py` computes `tau = dt / tau_inv` (default dt=0.001),
    baking the timestep into exported time constants. If the exported tau
    values seem wrong, check `examples/norse_nir_roundtrip.py` for details.

## 7. Import from SpikingJelly

```python
import torch
from spikingjelly.activation_based import neuron, layer, functional
from spikingjelly.activation_based.nir_exchange import export_to_nir

functional.set_step_mode(model, "s")
graph = export_to_nir(model, torch.randn(1, n_input), dt=1e-4)
network = from_nir(graph, dt=1e-4)
```

Verified: exact spike match across 27 configurations (1,350 steps, 0 mismatches).

## 8. The r-Encoding Problem

The `r` field in NIR neuron primitives means different things in different frameworks.
The same physical neuron produces different NIR `r` values:

| Framework | r encoding | dt for `from_nir()` |
|---|---|---|
| Sinabs | `r = 1.0` (fixed) | `dt=1.0` |
| snnTorch | `r = tau_mem / dt` (dt=1e-4) | `dt=1e-4` |
| Rockpool | `r = tau * exp(-dt/tau) / dt` | Match module's dt |
| Norse | `r = 1.0` | `dt=1.0` |
| SpikingJelly | tau has dt baked in | Match export dt |

Always match the dt to the exporting framework.

## 9. All 18 NIR Primitives

SC-NeuroCore maps every NIR primitive:

| NIR Primitive | SC-NeuroCore Node | Purpose |
|---|---|---|
| `Input` | `SCInputNode` | Graph entry point |
| `Output` | `SCOutputNode` | Graph exit point |
| `LIF` | `SCLIFNode` | Leaky integrate-and-fire |
| `IF` | `SCIFNode` | Integrate-and-fire (no leak) |
| `LI` | `SCLINode` | Leaky integrator (no threshold) |
| `I` | `SCIntegratorNode` | Pure integrator |
| `CubaLIF` | `SCCubaLIFNode` | Current-based LIF with synaptic filter |
| `CubaLI` | `SCCubaLINode` | Current-based leaky integrator |
| `Affine` | `SCAffineNode` | W @ x + b |
| `Linear` | `SCLinearNode` | W @ x |
| `Scale` | `SCScaleNode` | Element-wise scaling |
| `Threshold` | `SCThresholdNode` | Spike generation |
| `Flatten` | `SCFlattenNode` | Tensor reshape |
| `Delay` | `SCDelayNode` | Circular buffer delay |
| `Conv1d` | `SCConv1dNode` | 1D convolution |
| `Conv2d` | `SCConv2dNode` | 2D convolution |
| `SumPool2d` | `SCSumPool2dNode` | Spatial sum pooling |
| `AvgPool2d` | `SCAvgPool2dNode` | Average pooling |

## 10. From NIR to FPGA

The full pipeline: train in snnTorch → export to NIR → import into SC-NeuroCore →
compile to IR → emit SystemVerilog → synthesize for FPGA.

```python
from sc_neurocore.nir_bridge import from_nir
from sc_neurocore.compiler import compile_to_verilog
from sc_neurocore.compiler.equation_compiler import equation_to_fpga

# Step 1: Import NIR graph
network = from_nir(graph, dt=1e-4, reset_mode="subtract")

# Step 2: For individual neuron models, compile to Verilog
neuron, sv_code = equation_to_fpga(
    "dv/dt = (-v + I) / tau",
    threshold="v > 1.0", reset="v = 0.0",
    params={"tau": 20.0}, module_name="nir_lif",
)
with open("nir_lif.sv", "w") as f:
    f.write(sv_code)
print(f"Generated {len(sv_code)} chars of SystemVerilog")
```

## Runnable Demos

```bash
# Synthetic CubaLIF + recurrent (no extra deps)
python examples/nir_roundtrip_demo.py

# Norse weights (requires norse, torch)
python examples/norse_nir_roundtrip.py

# SpikingJelly weights (requires spikingjelly, torch)
python examples/spikingjelly_nir_roundtrip.py
```

## Further Reading

- [NIR Integration Guide](../guides/nir_integration.md) — full API reference and framework-specific details
- [NIR Bridge API](../api/nir_bridge.md) — auto-generated API docs
- [NIR Standard](https://neuroir.org/) — the NIR specification
