# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for quantisation observers

"""Tests for per-tensor and per-channel quantisation observers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sc_neurocore.qat.observers import (
    MinMaxObserver,
    PerChannelMinMaxObserver,
    _quant_bounds,
    fake_quantize,
)


class TestQuantBounds:
    def test_signed_bounds(self) -> None:
        assert _quant_bounds(8, unsigned=False) == (-128, 127)
        assert _quant_bounds(4, unsigned=False) == (-8, 7)

    def test_unsigned_bounds(self) -> None:
        assert _quant_bounds(8, unsigned=True) == (0, 255)
        assert _quant_bounds(4, unsigned=True) == (0, 15)

    def test_rejects_low_bits(self) -> None:
        with pytest.raises(ValueError, match="n_bits must be >= 2"):
            _quant_bounds(1, unsigned=False)


class TestMinMaxObserver:
    def test_tracks_running_range(self) -> None:
        obs = MinMaxObserver(8)
        obs.observe(torch.tensor([-1.0, 2.0]))
        obs.observe(torch.tensor([-3.0, 1.0]))
        assert obs.min_val.item() == -3.0
        assert obs.max_val.item() == 2.0

    def test_observe_returns_input(self) -> None:
        obs = MinMaxObserver(8)
        x = torch.randn(5)
        assert torch.equal(obs.observe(x), x)

    def test_symmetric_scale_scalar_and_zero_point(self) -> None:
        obs = MinMaxObserver(8, symmetric=True)
        obs.observe(torch.tensor([-4.0, 2.0]))
        scale, zp = obs.calculate_qparams()
        assert scale.shape == torch.Size([])
        # abs_max = 4, qmax = 127 -> scale = 4/127.
        assert scale.item() == pytest.approx(4.0 / 127)
        assert zp.item() == 0.0

    def test_affine_scale_and_zero_point(self) -> None:
        obs = MinMaxObserver(8, symmetric=False)
        obs.observe(torch.tensor([0.0, 4.0]))
        scale, zp = obs.calculate_qparams()
        # range 0..4 over 255 codes.
        assert scale.item() == pytest.approx(4.0 / 255, rel=1e-5)

    def test_calculate_before_observe_raises(self) -> None:
        obs = MinMaxObserver(8)
        with pytest.raises(RuntimeError, match="before any observation"):
            obs.calculate_qparams()

    def test_quantize_round_trips_within_tolerance(self) -> None:
        obs = MinMaxObserver(8, symmetric=True)
        x = torch.randn(1000)
        obs.observe(x)
        xq = obs.quantize(x)
        # 8-bit symmetric -> error bounded by ~scale.
        assert (xq - x).abs().max().item() < x.abs().max().item() / 100

    def test_zero_is_included_in_symmetric_range(self) -> None:
        obs = MinMaxObserver(8, symmetric=True)
        obs.observe(torch.tensor([2.0, 5.0]))  # all positive
        scale, _ = obs.calculate_qparams()
        # abs_max stays 5 (zero folded in), not the min 2.
        assert scale.item() == pytest.approx(5.0 / 127)


class TestPerChannelMinMaxObserver:
    def test_per_channel_scale_shape(self) -> None:
        obs = PerChannelMinMaxObserver(8, ch_axis=0)
        obs.observe(torch.randn(6, 8))
        scale, zp = obs.calculate_qparams()
        assert scale.shape == torch.Size([6])
        assert zp.shape == torch.Size([6])

    def test_per_channel_scales_track_channel_magnitude(self) -> None:
        obs = PerChannelMinMaxObserver(8, ch_axis=0, symmetric=True)
        w = torch.stack([torch.ones(8) * 0.1, torch.ones(8) * 1.0, torch.ones(8) * 10.0])
        obs.observe(w)
        scale, _ = obs.calculate_qparams()
        assert scale[2] > scale[1] > scale[0]

    def test_running_min_max_across_batches(self) -> None:
        obs = PerChannelMinMaxObserver(8, ch_axis=0)
        obs.observe(torch.zeros(3, 4))
        obs.observe(torch.stack([torch.full((4,), 5.0), torch.zeros(4), torch.full((4,), -2.0)]))
        assert obs.max_vals[0].item() == 5.0
        assert obs.min_vals[2].item() == -2.0

    def test_non_zero_ch_axis(self) -> None:
        obs = PerChannelMinMaxObserver(8, ch_axis=1)
        obs.observe(torch.randn(4, 6))
        scale, _ = obs.calculate_qparams()
        assert scale.shape == torch.Size([6])

    def test_calculate_before_observe_raises(self) -> None:
        obs = PerChannelMinMaxObserver(8)
        with pytest.raises(RuntimeError, match="before any observation"):
            obs.calculate_qparams()

    def test_quantize_uses_per_channel_scales(self) -> None:
        obs = PerChannelMinMaxObserver(8, ch_axis=0, symmetric=True)
        w = torch.stack([torch.ones(8) * 0.1, torch.ones(8) * 10.0])
        obs.observe(w)
        wq = obs.quantize(w)
        assert wq.shape == w.shape
        assert (wq - w).abs().max().item() < 0.5

    def test_per_channel_beats_per_tensor_on_skewed_channels(self) -> None:
        # Channels with very different scales: per-channel should quantise the
        # small channel far more accurately than a single per-tensor scale.
        w = torch.stack([torch.linspace(-0.1, 0.1, 32), torch.linspace(-10, 10, 32)])
        pc = PerChannelMinMaxObserver(4, ch_axis=0, symmetric=True)
        pt = MinMaxObserver(4, symmetric=True)
        pc.observe(w)
        pt.observe(w)
        err_pc = (pc.quantize(w)[0] - w[0]).abs().mean()
        err_pt = (pt.quantize(w)[0] - w[0]).abs().mean()
        assert err_pc < err_pt


class TestFakeQuantize:
    def test_broadcasts_scalar(self) -> None:
        x = torch.tensor([0.0, 1.0, 2.0])
        out = fake_quantize(x, torch.tensor(1.0), torch.tensor(0.0), n_bits=8, unsigned=False)
        assert torch.allclose(out, x)

    def test_clamps_to_grid(self) -> None:
        x = torch.tensor([1000.0])
        out = fake_quantize(x, torch.tensor(1.0), torch.tensor(0.0), n_bits=4, unsigned=False)
        # 4-bit signed max code = 7.
        assert out.item() == 7.0
