# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFusionEnergyEstimator from former test_sensor_fusion.py

"""Focused suite: TestFusionEnergyEstimator from former test_sensor_fusion.py."""

from __future__ import annotations

from sensor_fusion_support import *  # noqa: F403

class TestFusionEnergyEstimator:
    def test_basic_estimate(self):
        est = FusionEnergyEstimator(tech_node_nm=28)
        result = est.estimate(num_streams=4, num_channels=64, bitstream_length=256)
        assert result.total_uw > 0
        assert result.decorrelation_uw > 0
        assert result.attention_uw > 0
        assert result.routing_uw > 0

    def test_no_attention_lower_energy(self):
        est = FusionEnergyEstimator(tech_node_nm=28)
        with_attn = est.estimate(4, 64, 256, use_attention=True)
        without_attn = est.estimate(4, 64, 256, use_attention=False)
        assert without_attn.total_uw < with_attn.total_uw

    def test_sub_mw_for_small_config(self):
        est = FusionEnergyEstimator(tech_node_nm=7)
        result = est.estimate(
            num_streams=2, num_channels=4, bitstream_length=16, use_attention=False
        )
        assert result.total_mw < 1.0

    def test_scales_with_tech_node(self):
        est_7nm = FusionEnergyEstimator(tech_node_nm=7)
        est_28nm = FusionEnergyEstimator(tech_node_nm=28)
        r7 = est_7nm.estimate(4, 64, 256)
        r28 = est_28nm.estimate(4, 64, 256)
        assert r7.total_uw < r28.total_uw

    def test_total_mw_conversion(self):
        est = FusionEnergyEstimator()
        result = est.estimate(2, 8, 64)
        assert abs(result.total_mw - result.total_uw / 1000.0) < 1e-10
