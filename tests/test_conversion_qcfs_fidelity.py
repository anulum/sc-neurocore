# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — QCFS quantisation and training fidelity

"""Exercise the public QCFS forward lattice and actual optimizer gradients."""

from __future__ import annotations

import pytest
import torch

from sc_neurocore.conversion.qcfs import QCFSActivation


@pytest.mark.parametrize("steps", [1, 4, 8, 32])
@pytest.mark.parametrize("theta", [0.25, 1.0, 4.0])
def test_shifted_quantisation_grid(steps: int, theta: float) -> None:
    """Resolve both sides of every rounding boundary and both saturation tails."""
    values = torch.arange(-2, steps + 3, dtype=torch.float64)
    for offset in (-0.5001, -0.4999, 0.0, 0.4999, 0.5001):
        x = (values + offset) * theta / steps
        expected = (values + offset + 0.5).floor().clamp(0, steps) * theta / steps
        actual = QCFSActivation(T=steps, theta=theta).double()(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_input_and_learned_threshold_surrogate_derivatives() -> None:
    """Match the shifted continuous clipping support and threshold derivative."""
    layer = QCFSActivation(T=4, theta=2.0, learn_theta=True).double()
    x = torch.tensor([-1.0, -0.125, 0.625, 1.875, 3.0], dtype=torch.float64, requires_grad=True)
    y = layer(x)
    torch.testing.assert_close(y, torch.tensor([0.0, 0.0, 0.5, 2.0, 2.0], dtype=torch.float64))
    y.sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float64))
    assert layer.theta.grad is not None
    # Interior derivatives (0 - -.125)/2 and (.5 - .625)/2 cancel;
    # each of the two saturated upper outputs contributes one.
    torch.testing.assert_close(layer.theta.grad, torch.tensor(2.0, dtype=torch.float64))


def test_optimizer_changes_upstream_weight_and_reduces_loss() -> None:
    """Train an actual ANN step through QCFS instead of accepting a zero gradient."""
    linear = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        linear.weight.fill_(0.25)
    model = torch.nn.Sequential(linear, QCFSActivation(T=8).double())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
    x = torch.ones((8, 1), dtype=torch.float64)
    target = torch.full_like(x, 0.75)
    before = torch.nn.functional.mse_loss(model(x), target)
    optimizer.zero_grad()
    torch.autograd.backward(before)
    assert linear.weight.grad is not None
    torch.testing.assert_close(linear.weight.grad, torch.tensor([[-1.0]], dtype=torch.float64))
    optimizer.step()
    after = torch.nn.functional.mse_loss(model(x), target)
    assert float(after.detach()) < float(before.detach())
    torch.testing.assert_close(linear.weight.detach(), torch.tensor([[0.5]], dtype=torch.float64))


@pytest.mark.parametrize("steps", [0, -1, True, 1.5, 2**32])
def test_invalid_rate_grid_refused(steps: int) -> None:
    """Reject grids that cannot describe a positive simulation step count."""
    with pytest.raises(ValueError, match="positive integer"):
        QCFSActivation(T=steps)


@pytest.mark.parametrize("theta", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_invalid_constructor_threshold_refused(theta: float) -> None:
    """Reject nonfinite or nonpositive threshold scales before model use."""
    with pytest.raises(ValueError, match="finite and positive"):
        QCFSActivation(theta=theta)


@pytest.mark.parametrize("theta", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_trained_threshold_refused(theta: float) -> None:
    """Catch optimizer-corrupted thresholds when the public forward runs."""
    layer = QCFSActivation(learn_theta=True)
    with torch.no_grad():
        layer.theta.fill_(theta)
    with pytest.raises(ValueError, match="finite and positive"):
        layer(torch.ones(3))


def test_shifted_clamp_endpoint_derivatives() -> None:
    """Use zero input derivative at both shifted clipping endpoints."""
    layer = QCFSActivation(T=4, theta=2.0, learn_theta=True).double()
    x = torch.tensor([-0.25, 1.75], dtype=torch.float64, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.zeros_like(x))
    assert layer.theta.grad is not None
    torch.testing.assert_close(layer.theta.grad, torch.tensor(1.0, dtype=torch.float64))


def test_nonfinite_neighbours_keep_their_own_threshold_derivative() -> None:
    """A saturated infinite element contributes its own derivative, never a batch NaN.

    Upper saturation contributes one to the threshold derivative and lower
    saturation zero, also for infinite inputs; only a NaN input carries NaN,
    through its own NaN output.
    """
    layer = QCFSActivation(T=4, theta=2.0, learn_theta=True).double()
    x = torch.tensor(
        [-0.125, 0.625, float("inf"), -float("inf"), 3.0], dtype=torch.float64, requires_grad=True
    )
    y = layer(x)
    torch.testing.assert_close(y, torch.tensor([0.0, 0.5, 2.0, 0.0, 2.0], dtype=torch.float64))
    y.sum().backward()
    assert x.grad is not None and layer.theta.grad is not None
    torch.testing.assert_close(x.grad, torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(layer.theta.grad, torch.tensor(2.0, dtype=torch.float64))

    poisoned = QCFSActivation(T=4, theta=2.0, learn_theta=True).double()
    z = torch.tensor([0.625, float("nan")], dtype=torch.float64, requires_grad=True)
    poisoned(z).sum().backward()
    assert z.grad is not None and poisoned.theta.grad is not None
    torch.testing.assert_close(z.grad, torch.tensor([1.0, 0.0], dtype=torch.float64))
    assert bool(torch.isnan(poisoned.theta.grad))
