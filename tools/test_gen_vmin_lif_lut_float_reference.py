# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFloatReference from former test_gen_vmin_lif_lut.py

"""Focused suite: TestFloatReference from former test_gen_vmin_lif_lut.py."""

from __future__ import annotations

from gen_vmin_lif_lut_support import *  # noqa: F403

class TestFloatReference:
    def test_float_step_matches_pytorch_dynamics(self) -> None:
        # Manual computation of one Vmin_LIF step:
        # v = 0.5, x = 0.3
        # v = 0.5 * 0.75 + 0.3 = 0.675
        # z = 0.675 - (-5) = 5.675
        # softplus(5.675) = log(1 + e^5.675) ≈ 5.6785
        # v = -5 + 5.6785 ≈ 0.6785
        # 0.6785 < 1 → no spike
        cfg = VminLifConfig()
        v_next, spike = vmin_lif_step_float(0.5, 0.3, cfg)
        assert spike == 0
        assert v_next == pytest.approx(0.6785, abs=0.01)

    def test_float_step_threshold_crossing(self) -> None:
        cfg = VminLifConfig()
        v_next, spike = vmin_lif_step_float(0.9, 0.5, cfg)
        # 0.9 * 0.75 + 0.5 = 1.175 ≥ threshold(1.0) → spike, reset to v_reset=0
        # Then softplus floor: v = v_inf + softplus(0 - (-5)) = -5 + softplus(5) ≈ 0.0067
        assert spike == 1
        assert v_next == pytest.approx(0.0067, abs=0.005)
