# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for surrogate gradient functions

"""Tests for surrogate gradient functions."""

import pytest

torch = pytest.importorskip("torch")

from sc_neurocore.training.surrogate import (
    atan_surrogate,
    fast_sigmoid,
    superspike,
)


@pytest.mark.parametrize("fn", [fast_sigmoid, superspike, atan_surrogate])
class TestSurrogateGradients:
    def test_forward_is_heaviside(self, fn):
        x = torch.tensor([-1.0, -0.1, 0.0, 0.1, 1.0])
        out = fn(x)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
        assert torch.equal(out, expected)

    def test_backward_nonzero_everywhere(self, fn):
        x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        out = fn(x)
        out.sum().backward()
        assert x.grad is not None
        assert (x.grad.abs() > 0).all()

    def test_backward_peaks_near_threshold(self, fn):
        x_near = torch.tensor([0.01], requires_grad=True)
        x_far = torch.tensor([5.0], requires_grad=True)
        fn(x_near).backward()
        fn(x_far).backward()
        assert x_near.grad.abs().item() > x_far.grad.abs().item()

    def test_batch_shape(self, fn):
        x = torch.randn(32, 128, requires_grad=True)
        out = fn(x)
        assert out.shape == x.shape
        out.sum().backward()
        assert x.grad.shape == x.shape


def test_fast_sigmoid_slope_effect():
    """Steeper slope -> higher peak gradient at threshold (x=0)."""
    x = torch.tensor([0.001], requires_grad=True)
    fast_sigmoid(x, slope=50.0).backward()
    grad_steep = x.grad.item()

    x2 = torch.tensor([0.001], requires_grad=True)
    fast_sigmoid(x2, slope=5.0).backward()
    grad_gentle = x2.grad.item()

    assert grad_steep > grad_gentle
