# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPerChannelMinMaxObserver from former test_qat_observers.py

"""Focused suite: TestPerChannelMinMaxObserver from former test_qat_observers.py."""

from __future__ import annotations

from tests.qat_observers_support import *  # noqa: F403

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
