# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (input_validation) from former test_to_torch_bridge.py

from __future__ import annotations

from to_torch_bridge_support import *  # noqa: F403


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
