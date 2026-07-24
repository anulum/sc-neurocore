# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPACTActivation from former test_qat_pact.py

"""Focused suite: TestPACTActivation from former test_qat_pact.py."""

from __future__ import annotations

from tests.qat_pact_support import *  # noqa: F403


class TestPACTActivation:
    def test_output_within_clip(self) -> None:
        act = PACTActivation(n_bits=4, alpha_init=2.0)
        x = torch.linspace(-3, 5, 200)
        out = act(x)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 2.0 + 1e-6

    def test_output_quantised_to_levels(self) -> None:
        act = PACTActivation(n_bits=3, alpha_init=1.0)  # 7 positive levels
        x = torch.linspace(0, 1, 200)
        out = act(x)
        assert out.unique().numel() <= act.n_levels + 1

    def test_gradient_flows_to_alpha_and_input(self) -> None:
        act = PACTActivation(n_bits=4, alpha_init=2.0)
        x = (torch.randn(100) * 3).requires_grad_(True)  # leaf
        out = act(x)
        out.sum().backward()
        assert x.grad is not None
        assert act.alpha.grad is not None

    def test_alpha_is_parameter(self) -> None:
        act = PACTActivation(n_bits=4)
        assert isinstance(act.alpha, torch.nn.Parameter)

    def test_rejects_low_bits(self) -> None:
        with pytest.raises(ValueError, match="n_bits must be >= 2"):
            PACTActivation(n_bits=1)

    def test_quantize_returns_codes_and_scale(self) -> None:
        act = PACTActivation(n_bits=4, alpha_init=2.0)
        codes, scale = act.quantize(torch.linspace(-1, 4, 50))
        assert codes.dtype == torch.int32
        assert codes.min() >= 0
        assert codes.max() <= act.n_levels
        assert scale.item() == pytest.approx(2.0 / act.n_levels)

    def test_extra_repr(self) -> None:
        act = PACTActivation(n_bits=8, alpha_init=6.0)
        r = act.extra_repr()
        assert "n_bits=8" in r
        assert "6.000" in r

    def test_learns_to_shrink_alpha_on_bounded_data(self) -> None:
        # If all activations are small, the alpha gradient should push it down.
        act = PACTActivation(n_bits=4, alpha_init=10.0)
        opt = torch.optim.SGD(act.parameters(), lr=0.5)
        x = torch.rand(256) * 0.5  # data in [0, 0.5]
        start = act.alpha.item()
        for _ in range(20):
            opt.zero_grad()
            out = act(x)
            loss = ((out - x.clamp(0, 0.5)) ** 2).mean()
            loss.backward()
            opt.step()
        assert act.alpha.item() < start
