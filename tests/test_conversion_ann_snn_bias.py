# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ANN-to-SNN constant-current bias controls

"""Use analytical constant-drive controls on the actual public ANN converter."""

import numpy as np
import pytest
import torch

from sc_neurocore.conversion import QCFSActivation, convert


@pytest.mark.parametrize("steps", [8, 16, 64, 256])
def test_relu_bias_preserved_over_simulation_budget(steps: int) -> None:
    """A constant ANN bias must remain a constant per-step IF current."""
    linear = torch.nn.Linear(1, 1).double()
    with torch.no_grad():
        linear.weight.zero_()
        assert linear.bias is not None
        linear.bias.fill_(0.5)
    ann = torch.nn.Sequential(linear, torch.nn.ReLU())
    reference = ann(torch.zeros((3, 1), dtype=torch.float64)).detach().numpy()
    snn = convert(ann, T=steps)
    rates = snn.run(np.zeros((3, 1))) / steps
    np.testing.assert_array_equal(rates, reference)


@pytest.mark.parametrize("steps", [8, 16, 64, 256])
@pytest.mark.parametrize("theta", [1.0, 2.0, 4.0])
def test_qcfs_bias_normalised_with_layer_threshold(steps: int, theta: float) -> None:
    """Decode spike rates back to the source activation's physical output scale."""
    linear = torch.nn.Linear(1, 1).double()
    with torch.no_grad():
        linear.weight.zero_()
        assert linear.bias is not None
        linear.bias.fill_(0.5)
    ann = torch.nn.Sequential(linear, QCFSActivation(T=steps, theta=theta).double())
    reference = ann(torch.zeros((3, 1), dtype=torch.float64)).detach().numpy()
    snn = convert(ann, T=steps)
    decoded = snn.run(np.zeros((3, 1))) * theta / steps
    np.testing.assert_array_equal(decoded, reference)


@pytest.mark.parametrize("steps", [8, 16, 64, 256])
def test_two_layer_bias_and_weight_scaling_preserves_qcfs_outputs(steps: int) -> None:
    """Exercise bias normalisation across two distinct learned threshold scales."""
    first = torch.nn.Linear(2, 2).double()
    second = torch.nn.Linear(2, 2).double()
    with torch.no_grad():
        first.weight.zero_()
        assert first.bias is not None
        first.bias.copy_(torch.tensor([0.5, 1.0], dtype=torch.float64))
        second.weight.copy_(torch.tensor([[0.5, 0.25], [0.25, 0.5]], dtype=torch.float64))
        assert second.bias is not None
        second.bias.copy_(torch.tensor([0.25, 0.5], dtype=torch.float64))
    ann = torch.nn.Sequential(
        first,
        QCFSActivation(T=steps, theta=2.0).double(),
        second,
        QCFSActivation(T=steps, theta=4.0).double(),
    )
    reference = ann(torch.zeros((3, 2), dtype=torch.float64)).detach().numpy()
    snn = convert(ann, T=steps)
    decoded = snn.run(np.zeros((3, 2))) * 4.0 / steps
    np.testing.assert_array_equal(decoded, reference)
