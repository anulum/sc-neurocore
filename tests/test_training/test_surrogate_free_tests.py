# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_surrogate.py

"""Module-level tests from former test_surrogate.py."""

from __future__ import annotations

from tests.test_training.surrogate_support import *  # noqa: F403

def test_fast_sigmoid_slope_effect():
    """Steeper slope -> higher peak gradient at threshold (x=0)."""
    x = torch.tensor([0.001], requires_grad=True)
    fast_sigmoid(x, slope=50.0).backward()
    grad_steep = x.grad.item()

    x2 = torch.tensor([0.001], requires_grad=True)
    fast_sigmoid(x2, slope=5.0).backward()
    grad_gentle = x2.grad.item()

    assert grad_steep > grad_gentle
def test_sigmoid_surrogate_slope_controls_threshold_gradient() -> None:
    x_steep = torch.tensor([0.01], requires_grad=True)
    sigmoid_surrogate(x_steep, slope=20.0).backward()
    x_gentle = torch.tensor([0.01], requires_grad=True)
    sigmoid_surrogate(x_gentle, slope=2.0).backward()

    assert x_steep.grad.abs().item() > x_gentle.grad.abs().item()
def test_triangular_surrogate_width_controls_gradient_support() -> None:
    x_narrow = torch.tensor([0.8], requires_grad=True)
    triangular(x_narrow, width=0.5).backward()
    x_wide = torch.tensor([0.8], requires_grad=True)
    triangular(x_wide, width=2.0).backward()

    assert x_wide.grad.abs().item() > x_narrow.grad.abs().item()
