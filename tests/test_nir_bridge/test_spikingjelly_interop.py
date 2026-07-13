# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for NIR bridge (import, node mapping, execution)

"""Tests for nir_bridge: NIR graph → SC-NeuroCore network conversion."""

from typing import Any

import numpy as np
import pytest

nir = pytest.importorskip("nir")

from sc_neurocore.nir_bridge import from_nir


class TestSpikingJellyInterop:
    def test_spikingjelly_lif_roundtrip(self) -> None:
        """SpikingJelly LIFNode -> NIR -> SC-NeuroCore produces identical spikes."""
        pytest_mod = __import__("pytest")
        try:
            import torch
            from spikingjelly.activation_based import (  # type: ignore[import-not-found]  # optional dependency lacks stubs
                functional,
                layer,
                neuron,
            )
            from spikingjelly.activation_based.nir_exchange import (  # type: ignore[import-not-found]  # optional dependency lacks stubs
                export_to_nir,
            )
        except ImportError:
            pytest_mod.skip("spikingjelly not installed from git")

        torch.manual_seed(42)

        class Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = layer.Linear(4, 6)
                self.lif1 = neuron.LIFNode(tau=2.0)
                self.fc2 = layer.Linear(6, 2)
                self.lif2 = neuron.LIFNode(tau=2.0)

            def forward(self, x: Any) -> Any:
                x = self.lif1(self.fc1(x))
                x = self.lif2(self.fc2(x))
                return x

        model = Net()
        functional.set_step_mode(model, "s")
        graph = export_to_nir(model, torch.randn(1, 4), dt=1e-4)
        net = from_nir(graph, dt=1e-4)

        inp_t = torch.tensor([[5.0, 3.0, 1.0, 2.0]])
        inp_np = np.array([5.0, 3.0, 1.0, 2.0])
        functional.reset_net(model)

        mismatches = 0
        for _ in range(50):
            sj_out = model(inp_t).detach().numpy().flatten()
            sc_out = net.step({"x": inp_np})["output"]
            if not np.array_equal(sj_out, sc_out):
                mismatches += 1

        assert mismatches == 0, f"{mismatches}/50 mismatches"


# --- Graph parsing tests ---
