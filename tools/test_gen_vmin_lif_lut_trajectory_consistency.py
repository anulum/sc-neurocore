# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTrajectoryConsistency from former test_gen_vmin_lif_lut.py

"""Focused suite: TestTrajectoryConsistency from former test_gen_vmin_lif_lut.py."""

from __future__ import annotations

from gen_vmin_lif_lut_support import *  # noqa: F403


class TestTrajectoryConsistency:
    def test_constant_input_trajectory(self) -> None:
        cfg = VminLifConfig()
        lut = gen_softplus_lut(cfg.beta_v_inf, LUT_SIZE, LUT_RANGE)

        v_q = encode_q88(-3.0)
        v_f = -3.0
        spikes_q = []
        spikes_f = []
        for _ in range(50):
            v_q, sq = vmin_lif_step_q88(v_q, encode_q88(0.3), lut, cfg)
            v_f, sf = vmin_lif_step_float(v_f, 0.3, cfg)
            spikes_q.append(sq)
            spikes_f.append(sf)

        # Spike count should match within ±1 (due to LUT quantisation near threshold)
        assert abs(sum(spikes_q) - sum(spikes_f)) <= 1

    def test_zero_input_no_runaway(self) -> None:
        cfg = VminLifConfig()
        lut = gen_softplus_lut(cfg.beta_v_inf, LUT_SIZE, LUT_RANGE)
        v_q = 0
        for _ in range(100):
            v_q, _ = vmin_lif_step_q88(v_q, 0, lut, cfg)
        # With zero input, v should converge to a fixed point near 0 (not diverge)
        assert abs(decode_q88(v_q)) < 1.0
