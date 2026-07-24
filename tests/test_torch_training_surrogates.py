# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSurrogates from former test_torch_training.py

"""Focused suite: TestSurrogates from former test_torch_training.py."""

from __future__ import annotations

from tests.torch_training_support import *  # noqa: F403


class TestSurrogates:
    @pytest.mark.parametrize(
        "fn", [fast_sigmoid, superspike, atan_surrogate, sigmoid_surrogate, triangular]
    )
    def test_forward_is_heaviside(self, fn):
        x = torch.tensor([-1.0, -0.1, 0.1, 1.0])
        out = fn(x)
        assert torch.equal(out, torch.tensor([0.0, 0.0, 1.0, 1.0]))

    def test_straight_through_forward(self):
        x = torch.tensor([-1.0, 0.5])
        out = straight_through(x)
        assert torch.equal(out, torch.tensor([0.0, 1.0]))

    @pytest.mark.parametrize(
        "fn", [fast_sigmoid, superspike, atan_surrogate, sigmoid_surrogate, triangular]
    )
    def test_backward_nonzero(self, fn):
        x = torch.tensor([0.0], requires_grad=True)
        out = fn(x)
        out.backward()
        assert x.grad is not None
        assert x.grad.item() > 0

    def test_straight_through_backward(self):
        x = torch.tensor([0.5], requires_grad=True)
        out = straight_through(x)
        out.backward()
        assert x.grad.item() == 1.0

    @pytest.mark.parametrize(
        "fn", [fast_sigmoid, superspike, atan_surrogate, sigmoid_surrogate, triangular]
    )
    def test_gradient_peak_at_threshold(self, fn):
        """Surrogate gradient should peak near x=0 (threshold crossing)."""
        near = torch.tensor([0.01], requires_grad=True)
        far = torch.tensor([5.0], requires_grad=True)
        fn(near).backward()
        fn(far).backward()
        assert near.grad.abs().item() > far.grad.abs().item()

    def test_batch_surrogates(self):
        x = torch.randn(32, 128, requires_grad=True)
        out = atan_surrogate(x)
        assert out.shape == x.shape
        out.sum().backward()
        assert x.grad.shape == x.shape
