# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (projection_graph_rejects) from former test_to_torch_bridge.py

from __future__ import annotations

from to_torch_bridge_support import *  # noqa: F403


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


def test_network_to_torch_rejects_registered_projection_tensor_corruption() -> None:
    """Corrupted registered tensors report the affected projection parameter."""
    src = Population("LapicqueNeuron", 1, params={"tau": 5.0, "dt": 1.0}, label="src")
    out = Population("LapicqueNeuron", 1, params={"tau": 5.0, "dt": 1.0}, label="out")
    projection = Projection(src, out, weight=1.0)
    bridge = Network(src, out, projection).to_torch()
    bridge.register_parameter("proj_0_weight", None)

    with pytest.raises(TypeError, match="proj_0_weight"):
        bridge(torch.ones((1, 1, 1), dtype=torch.float32))


def test_network_to_torch_rejects_plastic_and_delayed_projections() -> None:
    src = Population("LapicqueNeuron", 1, params={"tau": 5.0, "dt": 1.0}, label="src")
    out = Population("LapicqueNeuron", 1, params={"tau": 5.0, "dt": 1.0}, label="out")

    plastic_projection = Projection(src, out, weight=1.0, plasticity="stdp")
    with pytest.raises(NotImplementedError, match="plastic projections"):
        Network(src, out, plastic_projection).to_torch()

    delayed_projection = Projection(src, out, weight=1.0, delay=1.0)
    with pytest.raises(NotImplementedError, match="delayed projections"):
        Network(src, out, delayed_projection).to_torch()
