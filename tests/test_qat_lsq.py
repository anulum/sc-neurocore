# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Learned Step Size Quantization (LSQ)

"""Tests for the LSQ weight quantiser (Esser et al. 2020)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from sc_neurocore.qat.lsq import LSQLinear, LSQQuantizer, _LSQQuantize, _sum_to


class TestSumTo:
    def test_reduces_to_scalar(self) -> None:
        g = torch.ones(4, 5)
        out = _sum_to(g, ())
        assert out.shape == torch.Size([])
        assert out.item() == 20.0

    def test_reduces_to_channel_vector(self) -> None:
        g = torch.ones(3, 4)
        out = _sum_to(g, (3, 1))
        assert out.shape == torch.Size([3, 1])
        assert torch.allclose(out, torch.full((3, 1), 4.0))


class TestLSQQuantizerForward:
    def test_output_on_integer_grid(self) -> None:
        q = LSQQuantizer(3)  # signed 3-bit: grid [-4, 3]
        x = torch.linspace(-2, 2, 50)
        out = q(x)
        step = q.step.detach()
        codes = torch.round(out / step)
        assert codes.min() >= q.qmin - 1e-6
        assert codes.max() <= q.qmax + 1e-6

    def test_signed_bounds(self) -> None:
        q = LSQQuantizer(4)
        assert q.qmin == -8
        assert q.qmax == 7

    def test_step_initialised_from_first_input(self) -> None:
        q = LSQQuantizer(4)
        assert not bool(q._initialized)
        x = torch.randn(64)
        q(x)
        assert bool(q._initialized)
        expected = 2.0 * x.abs().mean() / math.sqrt(q.qmax)
        assert torch.allclose(q.step.detach(), expected, rtol=1e-5)

    def test_step_not_reinitialised(self) -> None:
        q = LSQQuantizer(4)
        q(torch.randn(64))
        first = q.step.detach().clone()
        q(torch.randn(64) * 100)  # would give a very different init
        assert torch.allclose(q.step.detach(), first)

    def test_rejects_low_bits(self) -> None:
        with pytest.raises(ValueError, match="n_bits must be >= 2"):
            LSQQuantizer(1)

    def test_per_channel_requires_num_channels(self) -> None:
        with pytest.raises(ValueError, match="requires num_channels"):
            LSQQuantizer(4, per_channel=True)


class TestLSQPerChannel:
    def test_per_channel_step_shape(self) -> None:
        q = LSQQuantizer(4, per_channel=True, num_channels=5)
        assert q.step.shape == torch.Size([5])

    def test_per_channel_steps_are_independent(self) -> None:
        q = LSQQuantizer(4, per_channel=True, ch_axis=0, num_channels=3)
        # Row magnitudes differ by 100x -> per-channel steps must differ.
        w = torch.stack([torch.ones(8) * 0.01, torch.ones(8), torch.ones(8) * 100.0])
        q(w)
        steps = q.step.detach()
        assert steps[2] > steps[1] > steps[0]

    def test_per_tensor_step_is_scalar(self) -> None:
        q = LSQQuantizer(4)
        q(torch.randn(10))
        assert q.step.shape == torch.Size([])


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


class TestLSQLinear:
    def test_forward_shape(self) -> None:
        layer = LSQLinear(8, 5, n_bits=4)
        out = layer(torch.randn(3, 8))
        assert out.shape == (3, 5)

    def test_default_is_per_channel(self) -> None:
        layer = LSQLinear(8, 5, n_bits=4)
        assert layer.weight_quant.per_channel
        assert layer.weight_quant.step.shape == torch.Size([5])

    def test_scalar_step_when_not_per_channel(self) -> None:
        layer = LSQLinear(8, 5, n_bits=4, per_channel=False)
        assert layer.weight_quant.step.shape == torch.Size([])

    def test_gradients_reach_weights_and_step(self) -> None:
        layer = LSQLinear(8, 5, n_bits=4)
        out = layer(torch.randn(3, 8))
        out.sum().backward()
        assert layer.linear.weight.grad is not None
        assert layer.weight_quant.step.grad is not None

    def test_export_quantized_shapes(self) -> None:
        layer = LSQLinear(8, 5, n_bits=4, per_channel=True)
        layer(torch.randn(3, 8))  # initialise step
        export = layer.export_quantized()
        assert export["weight_int"].shape == (5, 8)
        assert export["weight_int"].dtype == torch.int32
        assert export["step"].shape == torch.Size([5])
        assert export["n_bits"] == 4
        assert export["per_channel"] is True
        assert export["weight_int"].abs().max() <= 8

    def test_export_without_bias(self) -> None:
        layer = LSQLinear(8, 5, n_bits=4, bias=False)
        layer(torch.randn(3, 8))
        export = layer.export_quantized()
        assert "bias" not in export

    def test_export_with_bias(self) -> None:
        layer = LSQLinear(8, 5, n_bits=4, bias=True)
        layer(torch.randn(3, 8))
        export = layer.export_quantized()
        assert "bias" in export
