# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRoundSTE from former test_qat_pact.py

"""Focused suite: TestRoundSTE from former test_qat_pact.py."""

from __future__ import annotations

from tests.qat_pact_support import *  # noqa: F403


class TestRoundSTE:
    def test_forward_rounds(self) -> None:
        x = torch.tensor([0.2, 0.8, -0.4])
        assert torch.allclose(_round_ste(x), torch.tensor([0.0, 1.0, -0.0]))

    def test_backward_is_identity(self) -> None:
        x = torch.tensor([0.3, 1.7], requires_grad=True)
        _round_ste(x).sum().backward()
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x.grad))
