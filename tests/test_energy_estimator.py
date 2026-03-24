# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for pre-silicon energy estimator

"""Tests for the FPGA energy estimation module."""

from __future__ import annotations

from sc_neurocore.energy.estimator import estimate, EnergyReport
from sc_neurocore.energy.fpga_models import TARGETS


class TestEstimate:
    def test_basic_estimate(self):
        report = estimate([(4, 2)], target="ice40")
        assert isinstance(report, EnergyReport)
        assert report.total_luts > 0
        assert report.total_dynamic_power_mw > 0

    def test_multi_layer(self):
        report = estimate([(8, 4), (4, 2)], target="ice40")
        assert len(report.layers) == 2
        assert report.total_latency_cycles > 0

    def test_small_network_fits_ice40(self):
        report = estimate([(4, 2)], target="ice40", bitstream_length=64)
        assert report.fits_on_target
        assert report.utilization_pct < 100

    def test_large_network_exceeds_ice40(self):
        report = estimate([(784, 128), (128, 10)], target="ice40")
        assert not report.fits_on_target
        assert report.utilization_pct > 100

    def test_artix7_more_capacity(self):
        layers = [(16, 8), (8, 4)]
        report_ice = estimate(layers, target="ice40")
        report_art = estimate(layers, target="artix7")
        assert report_art.utilization_pct < report_ice.utilization_pct

    def test_event_driven_less_power(self):
        layers = [(16, 8)]
        clock = estimate(layers, target="ice40", event_driven=False)
        event = estimate(layers, target="ice40", event_driven=True)
        assert event.total_dynamic_power_mw < clock.total_dynamic_power_mw

    def test_longer_bitstream_more_latency(self):
        short = estimate([(8, 4)], target="ice40", bitstream_length=64)
        long = estimate([(8, 4)], target="ice40", bitstream_length=1024)
        assert long.total_latency_cycles > short.total_latency_cycles

    def test_energy_scales_with_power_and_latency(self):
        report = estimate([(8, 4)], target="ice40")
        assert report.energy_per_inference_nj > 0

    def test_summary_string(self):
        report = estimate([(4, 2)], target="ice40")
        s = report.summary()
        assert "LUTs" in s
        assert "mW" in s
        assert "nJ" in s

    def test_all_targets_supported(self):
        for target in TARGETS:
            report = estimate([(4, 2)], target=target)
            assert report.total_luts > 0

    def test_unknown_target_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown target"):
            estimate([(4, 2)], target="nonexistent")

    def test_no_infra(self):
        with_infra = estimate([(4, 2)], target="ice40", include_infra=True)
        without = estimate([(4, 2)], target="ice40", include_infra=False)
        assert without.total_luts < with_infra.total_luts

    def test_bram_for_large_layers(self):
        small = estimate([(4, 2)], target="ice40")
        large = estimate([(100, 100)], target="ice40")
        assert large.total_bram_kb > small.total_bram_kb
