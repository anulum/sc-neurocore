#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# SC-NeuroCore -- NIR roundtrip demo: CubaLIF + recurrent connections
#
# Usage:
#   pip install sc-neurocore nir
#   python examples/nir_roundtrip_demo.py

"""Demonstrate NIR roundtrip: build graph -> from_nir() -> run -> to_nir() -> verify.

Tests CubaLIF and recurrent connections as requested by Jens Pedersen (NIR).
"""

import numpy as np

try:
    import nir
except ImportError:
    raise ImportError("pip install nir")

from sc_neurocore.nir_bridge import from_nir, to_nir


def build_recurrent_cubalif_graph():
    """Build a NIR graph with CubaLIF + recurrent Linear feedback."""
    np.random.seed(42)
    nodes = {
        "input": nir.Input(input_type={"input": np.array([4])}),
        "affine": nir.Affine(
            weight=np.random.randn(6, 4).astype(np.float32) * 0.5,
            bias=np.zeros(6, dtype=np.float32),
        ),
        "lif": nir.CubaLIF(
            tau_syn=np.full(6, 5.0, dtype=np.float32),
            tau_mem=np.full(6, 10.0, dtype=np.float32),
            r=np.full(6, 0.8, dtype=np.float32),
            v_leak=np.full(6, -0.1, dtype=np.float32),
            v_threshold=np.ones(6, dtype=np.float32),
            w_in=np.full(6, 1.2, dtype=np.float32),
            v_reset=np.full(6, -0.25, dtype=np.float32),
        ),
        "rec": nir.Linear(
            weight=np.random.randn(6, 6).astype(np.float32) * 0.1,
        ),
        "readout": nir.Affine(
            weight=np.random.randn(2, 6).astype(np.float32) * 0.3,
            bias=np.zeros(2, dtype=np.float32),
        ),
        "output": nir.Output(output_type={"output": np.array([2])}),
    }
    edges = [
        ("input", "affine"),
        ("affine", "lif"),
        ("lif", "rec"),  # recurrent: lif -> rec -> lif
        ("rec", "lif"),  # back edge (cycle)
        ("lif", "readout"),
        ("readout", "output"),
    ]
    return nir.NIRGraph(nodes=nodes, edges=edges)


def main():
    print("SC-NeuroCore NIR Roundtrip Demo")
    print("=" * 50)

    # 1. Build NIR graph
    graph = build_recurrent_cubalif_graph()
    print("\n1. Built NIR graph:")
    print(f"   Nodes: {sorted(graph.nodes.keys())}")
    print(f"   Edges: {graph.edges}")
    print(
        f"   CubaLIF tau_syn={graph.nodes['lif'].tau_syn[0]:.1f}, "
        f"tau_mem={graph.nodes['lif'].tau_mem[0]:.1f}"
    )
    print("   Recurrent: lif -> rec -> lif (cycle)")

    # 2. Import into SC-NeuroCore
    net = from_nir(graph, dt=1.0)
    print("\n2. Imported into SC-NeuroCore:")
    print(f"   {len(net.topo_order)} nodes in execution order")
    print(f"   Recurrent connections: {len(net._recurrent_map)}")
    delay_nodes = [n for n in net.nodes if n.startswith("_delay_")]
    if delay_nodes:
        print(f"   Delay nodes inserted: {delay_nodes}")

    # 3. Run simulation
    input_data = np.array([8.0, 6.0, 4.0, 3.0])
    n_steps = 200
    results = net.run({"input": input_data}, steps=n_steps)
    spike_counts = np.array([r.sum() for r in results["output"]])
    total_spikes = spike_counts.sum()
    print(f"\n3. Simulation ({n_steps} steps, input={input_data}):")
    print(f"   Total output spikes: {total_spikes:.0f}")
    print(f"   Output per step (first 10): {spike_counts[:10]}")

    # 4. Export back to NIR
    graph_out = to_nir(net)
    print("\n4. Exported back to NIR:")
    print(f"   Nodes: {sorted(graph_out.nodes.keys())}")
    print(f"   Edges: {graph_out.edges}")

    # 5. Verify roundtrip
    print("\n5. Roundtrip verification:")
    assert set(graph_out.nodes.keys()) == set(graph.nodes.keys()), "Node mismatch!"
    print("   Node names match: OK")

    assert len(graph_out.edges) == len(graph.edges), "Edge count mismatch!"
    print(f"   Edge count matches: OK ({len(graph_out.edges)} edges)")

    for name in graph.nodes:
        assert isinstance(graph_out.nodes[name], type(graph.nodes[name])), (
            f"Type mismatch for {name}"
        )
    print("   All node types match: OK")

    # Verify ALL CubaLIF parameters (including r, v_leak, v_reset)
    orig = graph.nodes["lif"]
    exported = graph_out.nodes["lif"]
    np.testing.assert_allclose(exported.tau_syn, orig.tau_syn)
    np.testing.assert_allclose(exported.tau_mem, orig.tau_mem)
    np.testing.assert_allclose(exported.r, orig.r)
    np.testing.assert_allclose(exported.v_leak, orig.v_leak)
    np.testing.assert_allclose(exported.v_threshold, orig.v_threshold)
    np.testing.assert_allclose(exported.w_in, orig.w_in)
    np.testing.assert_allclose(exported.v_reset, orig.v_reset)
    print("   CubaLIF ALL 7 parameters match: OK")

    # Verify full edge-set equality (not just count)
    assert set(graph_out.edges) == set(graph.edges), (
        f"Edge set mismatch!\n  Expected: {set(graph.edges)}\n  Got: {set(graph_out.edges)}"
    )
    print("   Full edge set matches: OK")

    # 6. File roundtrip
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".nir", delete=False) as f:
        path = f.name
    to_nir(net, path=path)
    graph_reload = nir.read(path)
    assert set(graph_reload.nodes.keys()) == set(graph.nodes.keys())
    assert set(graph_reload.edges) == set(graph.edges)
    import os

    os.unlink(path)
    print("   File save/load full roundtrip: OK")

    print(f"\n{'=' * 50}")
    print("ALL TESTS PASS -- NIR roundtrip with CubaLIF + recurrent connections verified.")


if __name__ == "__main__":
    main()
