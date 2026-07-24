# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPACTClip from former test_qat_pact.py

"""Focused suite: TestPACTClip from former test_qat_pact.py."""

from __future__ import annotations

from tests.qat_pact_support import *  # noqa: F403


class TestPACTClip:
    def test_clips_to_zero_alpha(self) -> None:
        x = torch.tensor([-1.0, 0.5, 3.0])
        alpha = torch.tensor(2.0)
        out = _PACTClip.apply(x, alpha)
        assert torch.allclose(out, torch.tensor([0.0, 0.5, 2.0]))

    def test_value_gradient_passes_inside_range(self) -> None:
        x = torch.tensor([-1.0, 0.5, 1.5, 3.0], requires_grad=True)
        alpha = torch.tensor(2.0, requires_grad=True)
        out = _PACTClip.apply(x, alpha)
        out.sum().backward()
        # Inside [0, alpha]: gradient passes; outside: zero.
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.tensor([0.0, 1.0, 1.0, 0.0]))

    def test_alpha_gradient_counts_saturated_inputs(self) -> None:
        x = torch.tensor([-1.0, 0.5, 3.0, 4.0], requires_grad=True)
        alpha = torch.tensor(2.0, requires_grad=True)
        out = _PACTClip.apply(x, alpha)
        out.sum().backward()
        # Two inputs exceed alpha -> d out / d alpha = 2.
        assert alpha.grad is not None
        assert alpha.grad.item() == pytest.approx(2.0)
