# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real snnTorch inference conversion tests

"""Compare actual framework inference, state and independent parameter ownership."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
snn = pytest.importorskip("snntorch")
pytest.importorskip("nirtorch")
from torch import nn

from sc_neurocore.nir_bridge import from_nir, from_snntorch, to_nir, to_snntorch
from tests.test_nir_bridge.snntorch_models import make_model as _model


@pytest.mark.parametrize("kind", ["leaky", "synaptic"])
@pytest.mark.parametrize("reset", ["subtract", "zero"])
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("disabled", [False, True])
def test_actual_multilayer_framework_in_both_directions(
    kind: str, reset: str, vector: bool, disabled: bool, tmp_path: Path
) -> None:
    """Compare actual framework spikes, states and parameter ownership in both directions."""
    model = _model(kind, reset, vector, disabled)
    # Warm the real source: conversion must snapshot definitions without altering live state.
    with torch.no_grad():
        model(torch.ones(4, dtype=torch.float64))
    saved = {name: value.clone() for name, value in model.state_dict().items()}
    network = from_snntorch(model, torch.zeros(4, dtype=torch.float64), dt=0.125)
    for name, value in saved.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    reference = deepcopy(model)
    for cell in reference:
        if isinstance(cell, (snn.Leaky, snn.Synaptic)):
            cell.reset_mem()
    path = tmp_path / "actual-framework.nir"
    to_nir(network, path)
    restored = from_nir(path)
    exported = to_snntorch(restored)
    assert sum(isinstance(m, (snn.Leaky, snn.Synaptic)) for m in exported.modules()) == 3
    xs = np.random.default_rng(54).uniform(-0.4, 2.0, (80, 4))
    spikes = 0
    prior_reset_steps = 0
    with torch.no_grad():
        for x in xs:
            if reset == "subtract":
                # Count states entering a real subsequent step, not a terminal
                # post-step overshoot whose correction never gets executed.
                prior_reset_steps += int(
                    any(
                        np.any(node.v > node.v_threshold)
                        for node in network.nodes.values()
                        if type(node).__name__ in ("SCLIFNode", "SCCubaLIFNode")
                    )
                )
            expected = reference(torch.from_numpy(x)).numpy()
            actual = network.step({network.input_nodes[0]: x})[network.output_nodes[0]]
            reloaded = restored.step({restored.input_nodes[0]: x})[restored.output_nodes[0]]
            foreign = exported(torch.from_numpy(x)).numpy()
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(reloaded, expected)
            np.testing.assert_array_equal(foreign, expected)
            spikes += int(actual.sum())
            for name, node in network.nodes.items():
                if type(node).__name__ in ("SCLIFNode", "SCCubaLIFNode"):
                    index = int(name.lstrip("_"))
                    np.testing.assert_allclose(
                        node.v, reference[index].mem.numpy(), rtol=0, atol=1e-12
                    )
                    if kind == "synaptic":
                        np.testing.assert_allclose(
                            node.i_syn, reference[index].syn.numpy(), rtol=0, atol=1e-12
                        )
    assert spikes > 30
    if reset == "subtract":
        assert prior_reset_steps > 0, "The real trace must execute prior_reset correction"
    # Ownership crosses both ways: exported trainable weights cannot alias NumPy source arrays.
    with torch.no_grad():
        exported[0].weight.add_(10)
    assert not np.array_equal(exported[0].weight.numpy(force=True), restored.nodes["_0"].weight)


def test_integrate_and_fire_and_linear_only_sources() -> None:
    """Run real IF and dense-only framework models through both public conversions."""
    for model in [
        nn.Sequential(nn.Linear(4, 3, dtype=torch.float64)),
        nn.Sequential(
            nn.Linear(4, 3, dtype=torch.float64),
            snn.Leaky(beta=1.0, init_hidden=True, reset_delay=False),
        ).double(),
    ]:
        source = from_snntorch(model, torch.ones(4, dtype=torch.float64), dt=0.25)
        result = to_snntorch(source)
        with torch.no_grad():
            for x in np.random.default_rng(42).uniform(-0.3, 2, (32, 4)):
                a = model(torch.from_numpy(x)).numpy()
                b = source.step({source.input_nodes[0]: x})[source.output_nodes[0]]
                c = result(torch.from_numpy(x)).numpy()
                np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
                np.testing.assert_allclose(a, c, rtol=0, atol=1e-12)
