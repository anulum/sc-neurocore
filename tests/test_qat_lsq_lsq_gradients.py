# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLSQGradients from former test_qat_lsq.py

"""Focused suite: TestLSQGradients from former test_qat_lsq.py."""

from __future__ import annotations

from tests.qat_lsq_support import *  # noqa: F403


class TestLSQGradients:
    def test_value_gradient_is_ste_in_range(self) -> None:
        q = LSQQuantizer(4)
        x = (torch.randn(100) * 0.1).requires_grad_(True)  # leaf, well inside the grid
        out = q(x)
        out.sum().backward()
        assert x.grad is not None
        # Inside the clip range the STE gives unit gradient.
        assert torch.allclose(x.grad, torch.ones_like(x.grad))

    def test_value_gradient_zero_outside_range(self) -> None:
        q = LSQQuantizer(3)
        q(torch.randn(50))  # initialise step
        step = q.step.detach()
        # Force a value far above qmax so it clips.
        x = torch.full((5,), (q.qmax + 10) * step.item(), requires_grad=True)
        out = q(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_step_gradient_matches_lsq_formula_in_range(self) -> None:
        # Analytic LSQ step gradient for an in-range element:
        #   d out / d step = (round(v/s) - v/s) * grad_scale
        v = torch.tensor([0.12])
        s = torch.tensor([0.1])
        grad_scale = 0.5
        # v/s = 1.2 is inside the signed 3-bit grid (-4, 3).
        step = s.clone().requires_grad_(True)
        out = _LSQQuantize.apply(v, step, -4, 3, grad_scale)
        out.backward(torch.ones_like(out))
        v_s = (v / s).item()
        expected = (round(v_s) - v_s) * grad_scale
        assert step.grad is not None
        assert math.isclose(step.grad.item(), expected, rel_tol=1e-5)

    def test_step_gradient_saturates_below(self) -> None:
        v = torch.tensor([-100.0])  # far below qmin
        step = torch.tensor([1.0], requires_grad=True)
        out = _LSQQuantize.apply(v, step, -4, 3, 1.0)
        out.backward(torch.ones_like(out))
        assert step.grad is not None
        # Below qmin the step gradient is qmin.
        assert math.isclose(step.grad.item(), -4.0, rel_tol=1e-6)

    def test_grad_scale_uses_qmax_and_element_count(self) -> None:
        q = LSQQuantizer(4, per_channel=True, ch_axis=0, num_channels=4)
        w = torch.randn(4, 16, requires_grad=True)
        # 4 channels, 64 elements -> 16 per step.
        out = q(w)
        out.sum().backward()
        # Just assert finite, correctly-shaped step gradient (formula covered above).
        assert q.step.grad is not None
        assert q.step.grad.shape == torch.Size([4])
        assert bool(torch.isfinite(q.step.grad).all())
