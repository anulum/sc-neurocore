# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the declarative Network to_torch bridge

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch

from sc_neurocore.network._torch_bridge import NetworkTorchBridge
from sc_neurocore.network import Network, Population, Projection
from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron
from sc_neurocore.training.surrogate import atan_surrogate_custom_op


def _all_to_all_topology(
    n_src: int, n_tgt: int, weight: float
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
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


def _projection_edge_values_from_bridge(
    bridge: Any, projection: Projection, name: str
) -> np.ndarray[Any, Any]:
    dense = getattr(bridge, f"{name}_weight").detach().cpu().numpy()
    values = []
    for src_idx in range(projection.source.n):
        for k in range(projection.indptr[src_idx], projection.indptr[src_idx + 1]):
            tgt_idx = projection.indices[k]
            values.append(dense[tgt_idx, src_idx])
    return np.asarray(values, dtype=np.float64)


def test_network_to_torch_matches_manual_numpy_semantics_for_lapicque_chain() -> None:
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


def test_network_to_torch_training_loss_decreases_on_simple_dataset() -> None:
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
        cast(Any, loss).backward()
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


def test_network_to_torch_rejects_unsupported_population_model() -> None:
    pop = Population("AdaptiveThresholdIFNeuron", 2)
    net = Network(pop)

    try:
        net.to_torch()
    except NotImplementedError as exc:
        assert "AdaptiveThresholdIFNeuron" in str(exc)
    else:
        raise AssertionError("Expected NotImplementedError for unsupported model")


def test_network_to_torch_validates_input_rank_and_dimension() -> None:
    pop = Population("LapicqueNeuron", 2, params={"tau": 5.0, "dt": 1.0}, label="src")
    net = Network(pop)
    bridge = net.to_torch()

    try:
        bridge(torch.zeros((2, 2), dtype=torch.float32))
    except ValueError as exc:
        assert "shape (T, batch, input_dim)" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-3D input tensor")

    try:
        bridge(torch.zeros((3, 1, 3), dtype=torch.float32))
    except ValueError as exc:
        assert "Expected input_dim=2" in str(exc)
    else:
        raise AssertionError("Expected ValueError for wrong input_dim")


def test_network_to_torch_validates_input_time_dtype_and_finiteness() -> None:
    pop = Population("LapicqueNeuron", 2, params={"tau": 5.0, "dt": 1.0}, label="src")
    bridge = Network(pop).to_torch()

    with pytest.raises(ValueError, match="timestep"):
        bridge(torch.zeros((0, 1, 2), dtype=torch.float32))

    with pytest.raises(ValueError, match="floating-point"):
        bridge(torch.zeros((1, 1, 2), dtype=torch.int64))

    bad = torch.zeros((1, 1, 2), dtype=torch.float32)
    bad[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        bridge(bad)


def test_network_to_torch_rejects_empty_populations() -> None:
    pop = Population("LapicqueNeuron", 0, params={"tau": 5.0, "dt": 1.0}, label="empty")

    with pytest.raises(ValueError, match="n > 0"):
        Network(pop).to_torch()


def test_network_torch_bridge_direct_empty_population_list_fails() -> None:
    """The bridge constructor rejects direct construction without populations."""
    with pytest.raises(ValueError, match="at least one population"):
        NetworkTorchBridge([], [])


def test_network_to_torch_rejects_projection_endpoint_outside_network() -> None:
    src = Population("LapicqueNeuron", 1, params={"tau": 5.0, "dt": 1.0}, label="src")
    out = Population("LapicqueNeuron", 1, params={"tau": 5.0, "dt": 1.0}, label="out")
    outsider = Population("LapicqueNeuron", 1, params={"tau": 5.0, "dt": 1.0}, label="outsider")
    projection = Projection(src, outsider, weight=1.0)

    with pytest.raises(ValueError, match="endpoints"):
        Network(src, out, projection).to_torch()


@pytest.mark.parametrize(
    ("topology", "message"),
    [
        (
            (
                np.array([0, 1], dtype=np.int64),
                np.array([0], dtype=np.int64),
                np.array([1.0], dtype=np.float64),
            ),
            "indptr",
        ),
        (
            (
                np.array([0, 2, 1], dtype=np.int64),
                np.array([0], dtype=np.int64),
                np.array([1.0], dtype=np.float64),
            ),
            "monotonic",
        ),
        (
            (
                np.array([0, 1, 1], dtype=np.int64),
                np.array([2], dtype=np.int64),
                np.array([1.0], dtype=np.float64),
            ),
            "indices",
        ),
        (
            (
                np.array([0, 1, 1], dtype=np.int64),
                np.array([0], dtype=np.int64),
                np.array([np.nan], dtype=np.float64),
            ),
            "finite",
        ),
        (
            (
                np.array([0.0, 1.0, 1.0], dtype=np.float64),
                np.array([0], dtype=np.int64),
                np.array([1.0], dtype=np.float64),
            ),
            "indptr",
        ),
        (
            (
                np.array([0, 1, 1], dtype=np.int64),
                np.array([0.0], dtype=np.float64),
                np.array([1.0], dtype=np.float64),
            ),
            "indices",
        ),
        (
            (
                np.array([0, 1, 1], dtype=np.int64),
                np.array([0], dtype=np.int64),
                np.array([1.0, 2.0], dtype=np.float64),
            ),
            "lengths",
        ),
    ],
)
def test_network_to_torch_rejects_malformed_projection_csr(
    topology: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]],
    message: str,
) -> None:
    src = Population("LapicqueNeuron", 2, params={"tau": 5.0, "dt": 1.0}, label="src")
    out = Population("LapicqueNeuron", 2, params={"tau": 5.0, "dt": 1.0}, label="out")
    projection = Projection(src, out, weight=0.0, topology=_all_to_all_topology(2, 2, 0.0))
    projection.indptr, projection.indices, projection.data = topology

    with pytest.raises(ValueError, match=message):
        Network(src, out, projection).to_torch()


def test_network_to_torch_rejects_duplicate_output_trace_labels() -> None:
    left = Population("LapicqueNeuron", 1, params={"tau": 5.0, "dt": 1.0}, label="out")
    right = Population("LapicqueNeuron", 1, params={"tau": 5.0, "dt": 1.0}, label="out")

    with pytest.raises(ValueError, match="labels"):
        Network(left, right).to_torch()


def test_network_to_torch_rejects_recurrent_graph_without_input_surface() -> None:
    """A closed recurrent graph cannot infer an external current input surface."""
    left = Population("LapicqueNeuron", 1, params={"tau": 5.0, "dt": 1.0}, label="left")
    right = Population("LapicqueNeuron", 1, params={"tau": 5.0, "dt": 1.0}, label="right")
    left_to_right = Projection(left, right, weight=1.0)
    right_to_left = Projection(right, left, weight=1.0)

    with pytest.raises(ValueError, match="at least one input population"):
        Network(left, right, left_to_right, right_to_left).to_torch()


def test_network_to_torch_return_traces_returns_output_label_trace_stack() -> None:
    pop = Population("LapicqueNeuron", 1, params={"tau": 2.0, "dt": 1.0}, label="out")
    net = Network(pop)
    bridge = net.to_torch()

    counts, traces = bridge(
        torch.tensor([[[2.5]], [[2.5]], [[0.0]]], dtype=torch.float32),
        return_traces=True,
    )

    assert counts.shape == (1, 1)
    assert list(traces) == ["out"]
    assert traces["out"].shape == (3, 1, 1)


def test_network_to_torch_supports_deterministic_stochastic_lif() -> None:
    pop = Population(
        StochasticLIFNeuron,
        2,
        params={
            "tau_mem": 5.0,
            "dt": 1.0,
            "resistance": 1.0,
            "noise_std": 0.0,
            "refractory_period": 0,
            "v_reset": 0.0,
            "v_rest": 0.0,
        },
        label="lif",
    )
    net = Network(pop)
    bridge = net.to_torch()

    counts = bridge(torch.tensor([[[3.0, 0.0]], [[3.0, 0.0]]], dtype=torch.float32))

    assert counts.shape == (1, 2)


def test_network_to_torch_rejects_registered_projection_tensor_corruption() -> None:
    """Corrupted registered tensors report the affected projection parameter."""
    src = Population("LapicqueNeuron", 1, params={"tau": 5.0, "dt": 1.0}, label="src")
    out = Population("LapicqueNeuron", 1, params={"tau": 5.0, "dt": 1.0}, label="out")
    projection = Projection(src, out, weight=1.0)
    bridge = Network(src, out, projection).to_torch()
    bridge.register_parameter("proj_0_weight", None)

    with pytest.raises(TypeError, match="proj_0_weight"):
        bridge(torch.ones((1, 1, 1), dtype=torch.float32))


def test_network_to_torch_rejects_stochastic_lif_with_noise() -> None:
    pop = Population(
        StochasticLIFNeuron,
        1,
        params={"noise_std": 0.1, "refractory_period": 0, "v_reset": 0.0, "v_rest": 0.0},
    )
    net = Network(pop)

    with pytest.raises(NotImplementedError, match="noise_std == 0.0"):
        net.to_torch()


def test_network_to_torch_rejects_stochastic_lif_with_refractory_period() -> None:
    pop = Population(
        StochasticLIFNeuron,
        1,
        params={"noise_std": 0.0, "refractory_period": 1, "v_reset": 0.0, "v_rest": 0.0},
    )
    net = Network(pop)

    with pytest.raises(NotImplementedError, match="refractory_period == 0"):
        net.to_torch()


def test_network_to_torch_rejects_stochastic_lif_with_entropy_source() -> None:
    pop = Population(
        StochasticLIFNeuron,
        1,
        params={
            "noise_std": 0.0,
            "refractory_period": 0,
            "v_reset": 0.0,
            "v_rest": 0.0,
            "entropy_source": object(),
        },
    )
    net = Network(pop)

    with pytest.raises(NotImplementedError, match="external entropy_source"):
        net.to_torch()


def test_network_to_torch_rejects_stochastic_lif_when_reset_differs_from_rest() -> None:
    pop = Population(
        StochasticLIFNeuron,
        1,
        params={"noise_std": 0.0, "refractory_period": 0, "v_reset": -1.0, "v_rest": 0.0},
    )
    net = Network(pop)

    with pytest.raises(NotImplementedError, match="v_reset == v_rest"):
        net.to_torch()


def test_network_to_torch_rejects_plastic_and_delayed_projections() -> None:
    src = Population("LapicqueNeuron", 1, params={"tau": 5.0, "dt": 1.0}, label="src")
    out = Population("LapicqueNeuron", 1, params={"tau": 5.0, "dt": 1.0}, label="out")

    plastic_projection = Projection(src, out, weight=1.0, plasticity="stdp")
    with pytest.raises(NotImplementedError, match="plastic projections"):
        Network(src, out, plastic_projection).to_torch()

    delayed_projection = Projection(src, out, weight=1.0, delay=1.0)
    with pytest.raises(NotImplementedError, match="delayed projections"):
        Network(src, out, delayed_projection).to_torch()
