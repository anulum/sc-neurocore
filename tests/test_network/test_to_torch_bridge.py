# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the declarative Network to_torch bridge

from __future__ import annotations

import numpy as np
import torch

from sc_neurocore.network import Network, Population, Projection
from sc_neurocore.training.surrogate import atan_surrogate_custom_op


def _all_to_all_topology(
    n_src: int, n_tgt: int, weight: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indptr = np.arange(0, n_src * n_tgt + 1, n_tgt, dtype=np.int64)
    indices = np.tile(np.arange(n_tgt, dtype=np.int64), n_src)
    data = np.full(n_src * n_tgt, weight, dtype=np.float64)
    return indptr, indices, data


def _manual_counts(
    inputs: torch.Tensor,
    proj_in_hid: Projection,
    proj_hid_out: Projection,
) -> torch.Tensor:
    inputs_np = inputs.detach().cpu().numpy()
    batch = inputs_np.shape[1]
    counts = np.zeros((batch, proj_hid_out.target.n), dtype=np.float32)

    for batch_idx in range(batch):
        src = Population("LapicqueNeuron", 2, params={"tau": 5.0, "dt": 1.0}, label="src")
        hid = Population("LapicqueNeuron", 3, params={"tau": 5.0, "dt": 1.0}, label="hid")
        out = Population("LapicqueNeuron", 2, params={"tau": 5.0, "dt": 1.0}, label="out")
        proj1 = Projection(src, hid, weight=0.0, topology=_all_to_all_topology(2, 3, 0.0))
        proj2 = Projection(hid, out, weight=0.0, topology=_all_to_all_topology(3, 2, 0.0))
        proj1.data[:] = proj_in_hid.data
        proj2.data[:] = proj_hid_out.data

        last_src = np.zeros(src.n, dtype=np.int8)
        last_hid = np.zeros(hid.n, dtype=np.int8)
        for t in range(inputs_np.shape[0]):
            src_spikes = src.step_all(inputs_np[t, batch_idx])
            hid_current = proj1.propagate(last_src)
            hid_spikes = hid.step_all(hid_current)
            out_current = proj2.propagate(last_hid)
            out_spikes = out.step_all(out_current)
            counts[batch_idx] += out_spikes.astype(np.float32)
            last_src = src_spikes
            last_hid = hid_spikes

    return torch.from_numpy(counts)


def _projection_edge_values_from_bridge(bridge, projection: Projection, name: str) -> np.ndarray:
    dense = getattr(bridge, f"{name}_weight").detach().cpu().numpy()
    values = []
    for src_idx in range(projection.source.n):
        for k in range(projection.indptr[src_idx], projection.indptr[src_idx + 1]):
            tgt_idx = projection.indices[k]
            values.append(dense[tgt_idx, src_idx])
    return np.asarray(values, dtype=np.float64)


def test_network_to_torch_matches_manual_numpy_semantics_for_lapicque_chain():
    src = Population("LapicqueNeuron", 2, params={"tau": 5.0, "dt": 1.0}, label="src")
    hid = Population("LapicqueNeuron", 3, params={"tau": 5.0, "dt": 1.0}, label="hid")
    out = Population("LapicqueNeuron", 2, params={"tau": 5.0, "dt": 1.0}, label="out")
    proj_in_hid = Projection(src, hid, weight=0.0, topology=_all_to_all_topology(2, 3, 0.0))
    proj_hid_out = Projection(hid, out, weight=0.0, topology=_all_to_all_topology(3, 2, 0.0))
    proj_in_hid.data[:] = np.array([1.8, 0.0, 1.8, 1.8, 1.8, 0.0], dtype=np.float64)
    proj_hid_out.data[:] = np.array([1.6, 0.0, 0.0, 1.6, 1.6, 1.6], dtype=np.float64)
    net = Network(src, hid, out, proj_in_hid, proj_hid_out)

    bridge = net.to_torch(surrogate_fn=atan_surrogate_custom_op)
    inputs = torch.tensor(
        [
            [[1.4, 0.0], [0.0, 1.4]],
            [[1.4, 0.0], [0.0, 1.4]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    bridge_counts = bridge(inputs)
    manual_counts = _manual_counts(inputs, proj_in_hid, proj_hid_out)

    assert torch.equal(bridge_counts, manual_counts)


def test_network_to_torch_training_loss_decreases_on_simple_dataset():
    src = Population("LapicqueNeuron", 2, params={"tau": 2.0, "dt": 1.0}, label="src")
    hid = Population("LapicqueNeuron", 4, params={"tau": 2.0, "dt": 1.0}, label="hid")
    out = Population("LapicqueNeuron", 2, params={"tau": 2.0, "dt": 1.0}, label="out")
    proj_in_hid = Projection(src, hid, weight=0.0, topology=_all_to_all_topology(2, 4, 0.0))
    proj_hid_out = Projection(hid, out, weight=0.0, topology=_all_to_all_topology(4, 2, 0.0))
    proj_in_hid.data[:] = np.array([1.6, 0.4, 1.2, 0.2, 0.2, 1.2, 0.4, 1.6], dtype=np.float64)
    proj_hid_out.data[:] = np.array([1.4, 0.2, 1.0, 0.1, 0.1, 1.0, 0.2, 1.4], dtype=np.float64)
    net = Network(src, hid, out, proj_in_hid, proj_hid_out)

    bridge = net.to_torch(surrogate_fn=atan_surrogate_custom_op)
    inputs = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [0.0, 3.0]],
            [[3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [0.0, 3.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [0.0, 3.0]],
            [[3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [0.0, 3.0]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    optimizer = torch.optim.Adam(bridge.parameters(), lr=0.1)
    initial_loss = None
    final_loss = None
    for _ in range(60):
        optimizer.zero_grad()
        counts = bridge(inputs)
        loss = torch.nn.functional.cross_entropy(counts, targets)
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())

    assert initial_loss is not None
    assert final_loss is not None
    assert final_loss < initial_loss

    bridge.sync_to_network()
    assert np.allclose(
        proj_in_hid.data, _projection_edge_values_from_bridge(bridge, proj_in_hid, "proj_0")
    )
    assert np.allclose(
        proj_hid_out.data, _projection_edge_values_from_bridge(bridge, proj_hid_out, "proj_1")
    )


def test_network_to_torch_rejects_unsupported_population_model():
    pop = Population("AdaptiveThresholdIFNeuron", 2)
    net = Network(pop)

    try:
        net.to_torch()
    except NotImplementedError as exc:
        assert "AdaptiveThresholdIFNeuron" in str(exc)
    else:
        raise AssertionError("Expected NotImplementedError for unsupported model")
