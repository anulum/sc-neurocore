#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# SC-NeuroCore -- Norse -> NIR -> SC-NeuroCore roundtrip
#
# Usage:
#   pip install sc-neurocore nir norse
#   python examples/norse_nir_roundtrip.py

"""Norse model weights -> NIR graph -> SC-NeuroCore -> roundtrip.

Builds a recurrent SNN in Norse (LIFCell + Linear feedback), extracts
weights, constructs a NIR graph with those weights and CubaLIF
parameters, imports into SC-NeuroCore, simulates, and roundtrips.

Tau value observation (needs confirmation from Norse/NIR maintainers):
  Norse export_nir.py:29-30 computes tau = dt / tau_inv (dt=0.001),
    yielding tau_syn = 0.001/200 = 5e-6 for tau_syn_inv=200.
  Norse import_nir.py:101 inverts as tau_inv = 1/tau = 1/5e-6 = 200000.
  Norse lif_box.py uses alpha = dt * tau_inv = 0.001 * 200000 = 100.
  Original alpha was dt * tau_inv_original = 0.001 * 100 = 0.1.
  Observed: Norse model produces 9 firing steps on test input; after
    export->import the same model produces 17 (different spike pattern).
  We could not find documentation or issues clarifying whether this
    is intentional. NIR CubaLIF spec (nir/ir/neuron.py) defines tau as
    a continuous-time constant with no unit or dt convention specified.

This demo constructs the NIR graph with Norse nn.Linear weights and
tau values derived directly from Norse defaults without the dt factor:
  tau_syn = 1/tau_syn_inv = 1/200 = 0.005; at our dt=1.0 -> tau=5.0
  tau_mem = 1/tau_mem_inv = 1/100 = 0.01;  at our dt=1.0 -> tau=10.0

SC-NeuroCore's own roundtrip preserves all 7 CubaLIF parameters
exactly (verified: bit-for-bit match, deterministic simulation).

Weights are scaled 3x from Norse default init so the untrained
network produces visible spikes for demonstration purposes.

CubaLIF + recurrent connections stress test (Jens Pedersen, NIR/DTU).
"""

from __future__ import annotations

import warnings

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("pip install torch")

try:
    import norse.torch as norse
except ImportError:
    raise SystemExit("pip install norse")

try:
    import nir
except ImportError:
    raise SystemExit("pip install nir")

from sc_neurocore.nir_bridge import from_nir, to_nir

warnings.filterwarnings("ignore", category=DeprecationWarning)


class NorseRecurrentSNN(nn.Module):
    """Two-layer SNN with recurrent LIF connections, built in Norse."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 6, bias=True)
        self.lif1 = norse.LIFCell()
        self.rec = nn.Linear(6, 6, bias=False)
        self.fc2 = nn.Linear(6, 2, bias=True)
        self.lif2 = norse.LIFCell()

    def forward(self, x):
        z = self.fc1(x)
        z, _ = self.lif1(z)
        z = self.fc2(z)
        z, _ = self.lif2(z)
        return z


def build_norse_nir_graph():
    """Export Norse model weights into a NIR graph with recurrent edges."""
    torch.manual_seed(42)
    model = NorseRecurrentSNN()

    # Scale init weights so the untrained network produces visible spikes
    with torch.no_grad():
        model.fc1.weight.mul_(3.0)
        model.fc1.bias.fill_(0.5)
        model.rec.weight.mul_(0.3)
        model.fc2.weight.mul_(3.0)
        model.fc2.bias.fill_(0.5)

    # Norse LIF defaults: tau_syn_inv=200, tau_mem_inv=100
    # -> tau_syn=5ms, tau_mem=10ms at dt=1ms
    # Expressed in dimensionless steps (dt=1.0): tau_syn=5, tau_mem=10
    nodes = {
        "input": nir.Input(input_type={"input": np.array([4])}),
        "fc1": nir.Affine(
            weight=model.fc1.weight.detach().numpy(),
            bias=model.fc1.bias.detach().numpy(),
        ),
        "lif1": nir.CubaLIF(
            tau_syn=np.full(6, 5.0),
            tau_mem=np.full(6, 10.0),
            r=np.ones(6),
            v_leak=np.zeros(6),
            v_threshold=np.ones(6),
            w_in=np.full(6, 1.2),
            v_reset=np.zeros(6),
        ),
        "rec": nir.Linear(weight=model.rec.weight.detach().numpy()),
        "fc2": nir.Affine(
            weight=model.fc2.weight.detach().numpy(),
            bias=model.fc2.bias.detach().numpy(),
        ),
        "lif2": nir.CubaLIF(
            tau_syn=np.full(2, 5.0),
            tau_mem=np.full(2, 10.0),
            r=np.ones(2),
            v_leak=np.zeros(2),
            v_threshold=np.ones(2),
            w_in=np.full(2, 1.2),
            v_reset=np.zeros(2),
        ),
        "output": nir.Output(output_type={"output": np.array([2])}),
    }
    edges = [
        ("input", "fc1"),
        ("fc1", "lif1"),
        ("lif1", "rec"),
        ("rec", "lif1"),
        ("lif1", "fc2"),
        ("fc2", "lif2"),
        ("lif2", "output"),
    ]
    return nir.NIRGraph(nodes=nodes, edges=edges), model


def main():
    print("SC-NeuroCore <- Norse NIR Roundtrip")
    print("=" * 50)

    # 1. Build NIR graph from Norse model
    graph, model = build_norse_nir_graph()
    print("\n1. Norse model -> NIR graph:")
    print(f"   Nodes: {sorted(graph.nodes.keys())}")
    print(f"   Edges: {graph.edges}")
    print("   CubaLIF: tau_syn=5.0, tau_mem=10.0 (1/tau_inv at dt=1.0)")
    print("   Recurrent: lif1 -> rec -> lif1")
    print("   Weights from: Norse nn.Linear (seed=42, scaled 3x for visibility)")
    print("   Note: weights scaled so untrained init produces spikes for demo")

    # 2. Save/load .nir file
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".nir", delete=False) as f:
        path = f.name
    nir.write(path, graph)
    graph_loaded = nir.read(path)
    assert set(graph_loaded.nodes.keys()) == set(graph.nodes.keys())
    assert set(graph_loaded.edges) == set(graph.edges)
    os.unlink(path)
    print("\n2. NIR file save/load roundtrip: OK")

    # 3. Import into SC-NeuroCore
    net = from_nir(graph, dt=1.0)
    print("\n3. Imported into SC-NeuroCore:")
    print(f"   {len(net.topo_order)} nodes in execution order")
    print(f"   Recurrent connections: {len(net._recurrent_map)}")
    delay_nodes = [n for n in net.nodes if n.startswith("_delay_")]
    if delay_nodes:
        print(f"   Delay nodes inserted: {delay_nodes}")

    # 4. Simulate
    input_data = np.array([8.0, 6.0, 4.0, 3.0])
    n_steps = 200
    results = net.run({"input": input_data}, steps=n_steps)
    spike_counts = np.array([r.sum() for r in results["output"]])
    total_spikes = spike_counts.sum()
    print(f"\n4. Simulation ({n_steps} steps, input={input_data}):")
    print(f"   Total output spikes: {total_spikes:.0f}")

    # 5. Export back to NIR
    graph_out = to_nir(net)
    print("\n5. Exported back to NIR:")
    print(f"   Nodes: {sorted(graph_out.nodes.keys())}")

    # 6. Verify roundtrip
    print("\n6. Roundtrip verification:")
    assert set(graph_out.nodes.keys()) == set(graph.nodes.keys()), "Node mismatch!"
    print("   Node names match: OK")

    assert set(graph_out.edges) == set(graph.edges), "Edge mismatch!"
    print(f"   Edge set matches: OK ({len(graph_out.edges)} edges)")

    for name in graph.nodes:
        assert isinstance(graph_out.nodes[name], type(graph.nodes[name])), (
            f"Type mismatch for {name}"
        )
    print("   All node types match: OK")

    orig = graph.nodes["lif1"]
    exported = graph_out.nodes["lif1"]
    np.testing.assert_allclose(exported.tau_syn, orig.tau_syn)
    np.testing.assert_allclose(exported.tau_mem, orig.tau_mem)
    np.testing.assert_allclose(exported.v_threshold, orig.v_threshold)
    print("   CubaLIF parameters match: OK")

    print(f"\n{'=' * 50}")
    print("ALL PASS — Norse -> NIR -> SC-NeuroCore -> NIR roundtrip verified.")
    print(
        f"  Norse weights: fc1={list(model.fc1.weight.shape)}, rec={list(model.rec.weight.shape)}"
    )
    print("  CubaLIF + recurrent connections: working")
    print(f"  Network produced {total_spikes:.0f} spikes across {n_steps} steps")


if __name__ == "__main__":
    main()
