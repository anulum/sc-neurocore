# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSurrogateGradients from former test_surrogate.py

"""Focused suite: TestSurrogateGradients from former test_surrogate.py."""

from __future__ import annotations

from tests.test_training.surrogate_support import *  # noqa: F403

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
