# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for PACT parameterised clipping activation

"""Tests for the PACT activation quantiser (Choi et al. 2018)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sc_neurocore.qat.pact import PACTActivation, _PACTClip, _round_ste


class TestRoundSTE:
    def test_forward_rounds(self) -> None:
        x = torch.tensor([0.2, 0.8, -0.4])
        assert torch.allclose(_round_ste(x), torch.tensor([0.0, 1.0, -0.0]))

    def test_backward_is_identity(self) -> None:
        x = torch.tensor([0.3, 1.7], requires_grad=True)
        _round_ste(x).sum().backward()
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x.grad))


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
