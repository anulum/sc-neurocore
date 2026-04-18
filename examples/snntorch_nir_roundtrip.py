#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore -- snnTorch -> NIR -> SC-NeuroCore roundtrip
#
# Usage:
#   pip install sc-neurocore nir
#   python examples/snntorch_nir_roundtrip.py
#
# snnTorch is NOT required — this demo constructs snnTorch-style NIR
# graphs directly, matching the exact encoding snnTorch uses internally.

"""snnTorch RSynaptic -> NIR -> SC-NeuroCore roundtrip demo.

Demonstrates:
1. Construct an NIR graph matching snnTorch RSynaptic export format
   (recurrent CubaLIF subgraph with dt=1e-4, subtract-reset)
2. Import into SC-NeuroCore with from_nir(dt=1e-4, reset_mode="subtract")
3. Run and verify spikes
4. Export back to NIR with to_nir()
5. Verify parameter fidelity on roundtrip

snnTorch conventions:
  - dt = 1e-4 (hardcoded in export_nir.py)
  - r = tau_mem / dt (compensating factor for Euler discretization)
  - w_in = tau_syn / dt (same)
  - beta = exp(-dt/tau_mem) ~= 1 - dt/tau_mem for small dt
  - alpha = exp(-dt/tau_syn) ~= 1 - dt/tau_syn
  - Subtract-reset: v = v - v_threshold (not v = v_reset)
  - RSynaptic exports as NIRGraph subgraph: Input->CubaLIF->Linear(w_rec)->CubaLIF->Output
"""

from __future__ import annotations

import numpy as np

import nir

from sc_neurocore.nir_bridge import from_nir, to_nir


def build_snntorch_rsynaptic_graph(
    n_input: int = 4,
    n_hidden: int = 6,
    beta: float = 0.8,
    alpha: float = 0.9,
    seed: int = 42,
) -> tuple[nir.NIRGraph, float]:
    """Build NIR graph matching snnTorch RSynaptic export format.

    Parameters match what snnTorch.export_nir.export_to_nir() produces
    for a Sequential(Linear, RSynaptic) model.
    """
    dt = 1e-4  # snnTorch hardcoded
    tau_mem = dt / (1 - beta)  # 5e-4
    tau_syn = dt / (1 - alpha)  # 1e-3
    r = tau_mem / dt  # 5.0 (Euler compensation)
    w_in = tau_syn / dt  # 10.0

    rng = np.random.RandomState(seed)

    # Recurrent weights (positive for reliable spiking in demo)
    w_rec = np.abs(rng.randn(n_hidden, n_hidden).astype(np.float32)) * 0.05

    # RSynaptic subgraph: Input -> CubaLIF <-> Linear(w_rec) -> Output
    sub_nodes = {
        "input": nir.Input(input_type={"input": np.array([n_hidden])}),
        "cubalif": nir.CubaLIF(
            tau_syn=np.full(n_hidden, tau_syn),
            tau_mem=np.full(n_hidden, tau_mem),
            r=np.full(n_hidden, r),
            v_leak=np.zeros(n_hidden),
            v_threshold=np.ones(n_hidden),
            w_in=np.full(n_hidden, w_in),
            v_reset=np.zeros(n_hidden),
        ),
        "w_rec": nir.Linear(weight=w_rec),
        "output": nir.Output(output_type={"output": np.array([n_hidden])}),
    }
    sub_edges = [
        ("input", "cubalif"),
        ("cubalif", "w_rec"),
        ("w_rec", "cubalif"),  # recurrent feedback (cycle)
        ("cubalif", "output"),
    ]
    rsynaptic = nir.NIRGraph(nodes=sub_nodes, edges=sub_edges)

    # Outer graph: Linear -> RSynaptic subgraph
    w_ff = np.abs(rng.randn(n_hidden, n_input).astype(np.float32)) * 0.5

    nodes = {
        "input": nir.Input(input_type={"input": np.array([n_input])}),
        "affine": nir.Affine(
            weight=w_ff,
            bias=np.zeros(n_hidden, dtype=np.float32),
        ),
        "rsynaptic": rsynaptic,
        "output": nir.Output(output_type={"output": np.array([n_hidden])}),
    }
    edges = [
        ("input", "affine"),
        ("affine", "rsynaptic"),
        ("rsynaptic", "output"),
    ]
    return nir.NIRGraph(nodes=nodes, edges=edges), dt


def main():
    print("snnTorch RSynaptic -> NIR -> SC-NeuroCore roundtrip demo")
    print("=" * 60)

    # 1. Build snnTorch-style graph
    graph, dt = build_snntorch_rsynaptic_graph(n_input=4, n_hidden=6)
    print("\n1. Built snnTorch RSynaptic graph:")
    print(f"   Nodes: {list(graph.nodes.keys())}")
    print(f"   Edges: {graph.edges}")
    print(f"   dt={dt} (snnTorch hardcoded)")

    # Check subgraph structure
    sub = graph.nodes["rsynaptic"]
    print(f"   Subgraph nodes: {list(sub.nodes.keys())}")
    print(f"   Subgraph edges: {sub.edges}")
    cuba = sub.nodes["cubalif"]
    print(
        f"   CubaLIF tau_mem={cuba.tau_mem[0]:.6f}, "
        f"tau_syn={cuba.tau_syn[0]:.6f}, r={cuba.r[0]:.1f}, w_in={cuba.w_in[0]:.1f}"
    )

    # 2. Import with snnTorch conventions
    network = from_nir(graph, dt=dt, reset_mode="subtract")
    print("\n2. Imported into SC-NeuroCore")
    print(f"   Execution order: {network.topo_order}")

    # 3. Run simulation
    n_steps = 200
    inp = np.array([2.0, 1.5, 1.0, 0.5])  # strong input
    total_spikes = 0
    spike_steps = []

    for t in range(n_steps):
        out = network.step({"input": inp})
        s = out["output"].sum()
        if s > 0:
            total_spikes += int(s)
            spike_steps.append(t)

    print(f"\n3. Simulation ({n_steps} steps, dt={dt}):")
    print(f"   Total output spikes: {total_spikes}")
    if spike_steps:
        print(f"   First spike at step {spike_steps[0]}, last at {spike_steps[-1]}")
    else:
        print("   (No spikes — try increasing input magnitude)")

    # 4. Export back to NIR
    graph_out = to_nir(network)
    print("\n4. Exported back to NIR:")
    print(f"   Nodes: {list(graph_out.nodes.keys())}")
    print(f"   Edges: {graph_out.edges}")

    # 5. Verify parameter roundtrip
    print("\n5. Parameter roundtrip verification:")
    orig_cuba = graph.nodes["rsynaptic"].nodes["cubalif"]
    # Find CubaLIF in exported graph
    exported_cuba = None
    for name, node in graph_out.nodes.items():
        if isinstance(node, nir.CubaLIF):
            exported_cuba = node
            break
        if isinstance(node, nir.NIRGraph):
            for sname, snode in node.nodes.items():
                if isinstance(snode, nir.CubaLIF):
                    exported_cuba = snode
                    break

    if exported_cuba is not None:
        checks = [
            ("tau_mem", orig_cuba.tau_mem, exported_cuba.tau_mem),
            ("tau_syn", orig_cuba.tau_syn, exported_cuba.tau_syn),
            ("r", orig_cuba.r, exported_cuba.r),
            ("w_in", orig_cuba.w_in, exported_cuba.w_in),
            ("v_threshold", orig_cuba.v_threshold, exported_cuba.v_threshold),
            ("v_leak", orig_cuba.v_leak, exported_cuba.v_leak),
        ]
        all_ok = True
        for param_name, orig, exported in checks:
            match = np.allclose(orig, exported)
            status = "MATCH" if match else "MISMATCH"
            if not match:
                all_ok = False
            print(f"   {param_name}: {status}")

        if all_ok:
            print("\n   All CubaLIF parameters roundtrip exactly.")
        else:
            print("\n   WARNING: Some parameters differ after roundtrip.")
    else:
        print("   Could not find CubaLIF in exported graph.")

    # 6. Save/load file roundtrip
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "snntorch_model.nir")
        nir.write(path, graph_out)
        graph_reload = nir.read(path)
        print(f"\n6. File roundtrip: wrote and re-read {path}")
        print(f"   Nodes after reload: {list(graph_reload.nodes.keys())}")

    print("\n" + "=" * 60)
    print("snnTorch RSynaptic roundtrip complete.")


if __name__ == "__main__":
    main()
