# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMinMaxObserver from former test_qat_observers.py

"""Focused suite: TestMinMaxObserver from former test_qat_observers.py."""

from __future__ import annotations

from tests.qat_observers_support import *  # noqa: F403


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
