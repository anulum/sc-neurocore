# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ASIC pre-synthesis estimate tests

"""Exercise deterministic area, power, timing, and scaling estimates."""

from __future__ import annotations

from sc_neurocore.asic_flow.asic_flow import PDKConfig, PDKType, PreSynthEstimator


class TestPreSynthEstimator:
    def test_basic_estimate(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        est = PreSynthEstimator.estimate(
            n_neurons=16,
            n_synapses=256,
            bitstream_width=256,
            n_aer_ports=4,
            pdk=pdk,
        )
        assert est.gate_count > 0
        assert est.area_um2 > 0
        assert est.dynamic_power_mw > 0
        assert est.max_frequency_mhz > 0

    def test_scaling_with_neurons(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        small = PreSynthEstimator.estimate(8, 64, 128, 2, pdk)
        large = PreSynthEstimator.estimate(128, 1024, 256, 16, pdk)
        assert large.gate_count > small.gate_count
        assert large.area_um2 > small.area_um2
        assert large.dynamic_power_mw > small.dynamic_power_mw

    def test_pdk_scaling(self) -> None:
        sky = PDKConfig.from_pdk_type(PDKType.SKY130)
        tsmc = PDKConfig.from_pdk_type(PDKType.TSMC28)
        est_sky = PreSynthEstimator.estimate(16, 256, 256, 4, sky)
        est_tsmc = PreSynthEstimator.estimate(16, 256, 256, 4, tsmc)
        # 28nm should have smaller area than 130nm
        assert est_tsmc.area_um2 < est_sky.area_um2

    def test_power_scaling(self) -> None:
        sky = PDKConfig.from_pdk_type(PDKType.SKY130)
        tsmc = PDKConfig.from_pdk_type(PDKType.TSMC28)
        est_sky = PreSynthEstimator.estimate(16, 256, 256, 4, sky)
        est_tsmc = PreSynthEstimator.estimate(16, 256, 256, 4, tsmc)
        # Lower voltage at 28nm → less dynamic power
        assert est_tsmc.dynamic_power_mw < est_sky.dynamic_power_mw
