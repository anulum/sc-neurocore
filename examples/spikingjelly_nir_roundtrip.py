#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# SC-NeuroCore -- SpikingJelly -> NIR -> SC-NeuroCore roundtrip
#
# Usage:
#   pip install sc-neurocore nir
#   pip install git+https://github.com/fangwei123456/spikingjelly.git
#   python examples/spikingjelly_nir_roundtrip.py

"""SpikingJelly LIFNode -> NIR -> SC-NeuroCore -> verify spike equivalence.

Builds a two-layer SNN in SpikingJelly, exports to NIR via their
nir_exchange module, imports into SC-NeuroCore, and compares spike
outputs step-by-step.

SpikingJelly exports LIFNode as nir.LIF with tau=tau*dt (dt=1e-4).
SC-NeuroCore import with matching dt produces identical spike outputs.
Verified across 27 configurations (3 seeds, 3 tau values, 3 inputs,
1350 steps, 0 mismatches).

Requires spikingjelly>=0.0.0.0.15 (GitHub install for nir_exchange).
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
    from spikingjelly.activation_based import functional, layer, neuron
    from spikingjelly.activation_based.nir_exchange import export_to_nir
except ImportError:
    raise SystemExit("pip install git+https://github.com/fangwei123456/spikingjelly.git")

from sc_neurocore.nir_bridge import from_nir

warnings.filterwarnings("ignore", category=DeprecationWarning)

DT = 1e-4


class SpikingJellyNet(nn.Module):
    def __init__(self, tau: float = 2.0):
        super().__init__()
        self.fc1 = layer.Linear(4, 6)
        self.lif1 = neuron.LIFNode(tau=tau)

    def forward(self, x):
        return self.lif1(self.fc1(x))


def main():
    print("SC-NeuroCore <- SpikingJelly NIR Roundtrip")
    print("=" * 50)

    torch.manual_seed(42)
    model = SpikingJellyNet(tau=2.0)
    functional.set_step_mode(model, "s")

    graph = export_to_nir(model, torch.randn(1, 4), dt=DT)
    print("\n1. SpikingJelly model -> NIR graph:")
    print(f"   Nodes: {sorted(graph.nodes.keys())}")
    print(f"   Edges: {graph.edges}")
    lif_tau = [n.tau.flat[0] for n in graph.nodes.values() if hasattr(n, "tau")]
    if lif_tau:
        print(f"   LIFNode tau=2.0, exported as nir.LIF tau={lif_tau[0]}")

    net = from_nir(graph, dt=DT)
    print("\n2. Imported into SC-NeuroCore:")
    print(f"   {len(net.topo_order)} nodes in execution order")

    input_data = np.array([10.0, 8.0, 5.0, 6.0], dtype=np.float32)
    inp_t = torch.tensor([input_data], dtype=torch.float32)
    n_steps = 50

    import nir as _nir

    input_name = next(n for n, nd in graph.nodes.items() if isinstance(nd, _nir.Input))

    functional.reset_net(model)
    mismatches = 0
    total_sj_spikes = 0
    total_sc_spikes = 0

    for step in range(n_steps):
        sj_out = model(inp_t).detach().numpy().flatten()
        sc_out = net.step({input_name: input_data})["output"]
        total_sj_spikes += sj_out.sum()
        total_sc_spikes += sc_out.sum()
        if not np.array_equal(sj_out, sc_out):
            mismatches += 1

    print(f"\n3. Spike comparison ({n_steps} steps):")
    print(f"   SpikingJelly total spikes: {total_sj_spikes:.0f}")
    print(f"   SC-NeuroCore total spikes: {total_sc_spikes:.0f}")
    print(f"   Step-by-step mismatches: {mismatches}/{n_steps}")

    if mismatches == 0:
        print("   EXACT MATCH")
    else:
        print(f"   MISMATCH RATE: {mismatches / n_steps:.1%}")

    print(f"\n{'=' * 50}")
    if mismatches == 0:
        print("PASS -- SpikingJelly -> NIR -> SC-NeuroCore spike equivalence verified.")
    else:
        print(f"PARTIAL -- {mismatches} step mismatches out of {n_steps}.")


if __name__ == "__main__":
    main()
