# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual Sinabs export through its compatible NIR profile

"""Exercise upstream Sinabs files without claiming exponential/Euler equivalence."""

from importlib import metadata, import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sinabs")
nir = pytest.importorskip("nir")
sl = import_module("sinabs.layers")
sinabs_to_nir = import_module("sinabs.nir").to_nir

from sc_neurocore.nir_bridge import from_nir, to_nir


def _model() -> Any:
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3),
        sl.LIFSqueeze(
            tau_mem=torch.full((3,), 10.0),
            spike_threshold=torch.full((3,), 0.7),
            batch_size=1,
            norm_input=False,
        ),
    )
    with torch.no_grad():
        model[0].weight.fill_(0.2)
        model[0].bias.fill_(0.1)
    return model


def test_coherent_legacy_profile_runs_real_export_and_reference_file_roundtrip(
    tmp_path: Path,
) -> None:
    """A real upstream export executes and retains the selected SC clock in HDF5."""
    assert metadata.version("nir") == "1.0.4"
    assert metadata.version("nirtorch") == "2.0.2"
    assert metadata.version("sinabs") == "3.1.3"
    model = _model()
    graph = sinabs_to_nir(model, torch.ones(1, 4))
    network = from_nir(graph, dt=1.0, reset_mode="subtract")
    assert network.nodes["1"].tau.shape == (3,)
    np.testing.assert_array_equal(network.nodes["0"].weight, model[0].weight.detach().numpy())
    path = tmp_path / "sinabs-reference.nir"
    to_nir(network, path)
    restored = from_nir(path)
    assert restored.dt == 1 and restored.reset_mode == "subtract"
    assert list(restored.nodes) == list(network.nodes)
    spikes = 0
    for x in np.random.default_rng(6).uniform(0.7, 2, (80, 4)):
        before = network.step({"input": x})["output"]
        after = restored.step({"input": x})["output"]
        np.testing.assert_array_equal(after, before)
        spikes += int(before.sum())
    assert spikes > 10


def test_real_sinabs_export_does_not_imply_matching_discrete_dynamics() -> None:
    """Expose the actual exponential/Euler difference rather than accepting 30% drift."""
    model = _model()
    graph = sinabs_to_nir(model, torch.ones(1, 4))
    model[1].reset_states()
    network = from_nir(graph, dt=1.0, reset_mode="subtract")
    foreign, reference = [], []
    with torch.no_grad():
        for _ in range(80):
            foreign.append(model(torch.ones(1, 4)).numpy().reshape(-1).copy())
            reference.append(network.step({"input": np.ones(4)})["output"].copy())
    assert not np.array_equal(foreign, reference)
    assert np.count_nonzero(foreign) > 0 and np.count_nonzero(reference) > 0
    assert any("reset rule" in note.detail for note in network.interchange_notes)


def test_legacy_nir_refuses_reset_voltage_it_cannot_write() -> None:
    """An old file schema cannot silently drop a changed reset potential."""
    graph = sinabs_to_nir(_model(), torch.ones(1, 4))
    network = from_nir(graph)
    network.nodes["1"].v_reset[:] = -0.25
    with pytest.raises(ValueError, match="cannot preserve nonzero reset potentials"):
        to_nir(network)
