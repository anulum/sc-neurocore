# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdditionalSurrogateContracts from former test_surrogate.py

"""Focused suite: TestAdditionalSurrogateContracts from former test_surrogate.py."""

from __future__ import annotations

from tests.test_training.surrogate_support import *  # noqa: F403


@pytest.mark.parametrize("fn", [sigmoid_surrogate, straight_through, triangular])
class TestAdditionalSurrogateContracts:
    def test_forward_is_heaviside(self, fn):
        x = torch.tensor([-1.0, -0.1, 0.0, 0.1, 1.0])
        out = fn(x)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
        assert torch.equal(out, expected)

    def test_backward_has_gradient_support(self, fn):
        x = torch.tensor([-0.5, 0.0, 0.5], requires_grad=True)
        fn(x).sum().backward()
        assert x.grad is not None
        assert (x.grad.abs() > 0).all()

    def test_batch_shape_is_preserved(self, fn):
        x = torch.randn(16, 64, requires_grad=True)
        out = fn(x)
        assert out.shape == x.shape
        out.sum().backward()
        assert x.grad.shape == x.shape
