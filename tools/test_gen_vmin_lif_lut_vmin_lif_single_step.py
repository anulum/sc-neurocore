# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVminLifSingleStep from former test_gen_vmin_lif_lut.py

"""Focused suite: TestVminLifSingleStep from former test_gen_vmin_lif_lut.py."""

from __future__ import annotations

from gen_vmin_lif_lut_support import *  # noqa: F403


class TestVminLifSingleStep:
    def test_zero_input_zero_state(self) -> None:
        cfg = VminLifConfig()
        lut = gen_softplus_lut(cfg.beta_v_inf, LUT_SIZE, LUT_RANGE)
        # v=0, x=0 → after decay: 0, then softplus floor with v_inf=-5
        # z = 0 - (-5) = 5 → softplus(5) ≈ 5.0067 → v_new = -5 + 5.0067 ≈ 0.0067
        v_next, spike = vmin_lif_step_q88(0, 0, lut, cfg)
        assert spike == 0
        assert decode_q88(v_next) == pytest.approx(0.0067, abs=0.01)

    def test_threshold_crossing_triggers_spike(self) -> None:
        cfg = VminLifConfig()
        lut = gen_softplus_lut(cfg.beta_v_inf, LUT_SIZE, LUT_RANGE)
        # v=0.9, x=0.5 → decay: 0.9*0.75=0.675, +0.5=1.175 > threshold(1.0)
        # JIT eval order: charge→threshold→reset→softplus
        # After reset: v=0. After softplus(0-(-5))=softplus(5)≈5.0067 → v=-5+5.0067≈0.0067
        v_q88 = encode_q88(0.9)
        x_q88 = encode_q88(0.5)
        v_next, spike = vmin_lif_step_q88(v_q88, x_q88, lut, cfg)
        assert spike == 1
        # v_next is v_reset(=0) passed through softplus floor → ~0.0067 in float
        # In Q8.8 that's encode_q88(0.0067) ≈ 1-3 depending on LUT
        assert 0 <= v_next <= 5  # bounded near 0 by softplus

    def test_subthreshold_no_spike(self) -> None:
        cfg = VminLifConfig()
        lut = gen_softplus_lut(cfg.beta_v_inf, LUT_SIZE, LUT_RANGE)
        v_next, spike = vmin_lif_step_q88(encode_q88(0.5), encode_q88(0.1), lut, cfg)
        assert spike == 0

    def test_softplus_floor_prevents_unbounded_negative(self) -> None:
        cfg = VminLifConfig()
        lut = gen_softplus_lut(cfg.beta_v_inf, LUT_SIZE, LUT_RANGE)
        # Even with strongly negative state, softplus floor should bound v
        v_q88 = encode_q88(-4.0)
        v_next, spike = vmin_lif_step_q88(v_q88, encode_q88(-1.0), lut, cfg)
        assert decode_q88(v_next) >= cfg.v_inf  # bounded below by v_inf
        assert spike == 0

    def test_charged_state_saturates_to_q88_max_before_spike(self) -> None:
        cfg = VminLifConfig(v_threshold=1e9)
        lut = gen_softplus_lut(cfg.beta_v_inf, LUT_SIZE, LUT_RANGE)
        v_next, spike = vmin_lif_step_q88(Q88_MAX, Q88_MAX, lut, cfg)

        assert spike == 1
        assert 0 <= v_next <= 5

    def test_charged_state_saturates_to_q88_min(self) -> None:
        cfg = VminLifConfig(v_inf=0.0)
        v_next, spike = vmin_lif_step_q88(Q88_MIN, Q88_MIN, [0], cfg)

        assert spike == 0
        assert v_next == 0

    def test_floor_saturates_to_q88_min(self) -> None:
        cfg = VminLifConfig(v_inf=-1e9)
        v_next, spike = vmin_lif_step_q88(Q88_MIN, Q88_MIN, [-1000], cfg)

        assert spike == 0
        assert v_next == Q88_MIN

    def test_floor_saturates_to_q88_max(self) -> None:
        cfg = VminLifConfig(v_inf=1e9, v_threshold=1e9)
        v_next, spike = vmin_lif_step_q88(0, 0, [1000], cfg)

        assert spike == 0
        assert v_next == Q88_MAX
