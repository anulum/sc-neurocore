# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (semantics_and_training) from former test_to_torch_bridge.py

from __future__ import annotations

from to_torch_bridge_support import *  # noqa: F403


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
