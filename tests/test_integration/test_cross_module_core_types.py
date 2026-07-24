# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCoreTypes from former test_cross_module.py

"""Focused suite: TestCoreTypes from former test_cross_module.py."""

from __future__ import annotations

from cross_module_support import *  # noqa: F403


class TestCoreTypes:
    """Verify shared type system works across modules."""

    def test_hardware_budget(self):
        from sc_neurocore.core.types import HardwareBudget

        b = HardwareBudget(max_luts=100_000)
        util = b.utilisation(luts=50_000)
        assert abs(util["luts"] - 0.5) < 1e-10

    def test_resource_report_meets_budget(self):
        from sc_neurocore.core.types import HardwareBudget, ResourceReport

        b = HardwareBudget(max_luts=100_000, max_power_mw=1000.0)
        r = ResourceReport(total_luts=50_000, total_power_mw=500.0)
        assert r.meets_budget(b)

    def test_resource_report_exceeds_budget(self):
        from sc_neurocore.core.types import HardwareBudget, ResourceReport

        b = HardwareBudget(max_luts=100_000, max_power_mw=1000.0)
        r = ResourceReport(total_luts=200_000, total_power_mw=500.0)
        assert not r.meets_budget(b)

    def test_layer_spec_estimation(self):
        from sc_neurocore.core.types import LayerSpec, DecorrelationStrategy

        ls = LayerSpec(
            layer_id="L0",
            neurons=64,
            bitstream_length=256,
            decorrelator=DecorrelationStrategy.SOBOL,
        )
        assert ls.estimate_luts() > 0
        assert ls.estimate_power_mw() > 0
        assert 0 < ls.estimate_accuracy() <= 1.0

    def test_estimate_network(self):
        from sc_neurocore.core.types import LayerSpec, estimate_network

        layers = [
            LayerSpec(layer_id="L0", neurons=32),
            LayerSpec(layer_id="L1", neurons=64),
        ]
        report = estimate_network(layers)
        assert report.total_luts > 0
        assert report.mean_accuracy > 0

    def test_layer_spec_deterministic(self):
        from sc_neurocore.core.types import LayerSpec, ComputeMode

        ls = LayerSpec(layer_id="L0", neurons=10, mode=ComputeMode.DETERMINISTIC)
        assert ls.estimate_accuracy() == 1.0

    def test_resource_report_exceeds_each_budget_dimension(self) -> None:
        # meets_budget returns on the first failing dimension, so each branch
        # needs a report that clears the earlier checks and breaches one.
        from sc_neurocore.core.types import HardwareBudget, ResourceReport

        power = ResourceReport(total_power_mw=2_000.0)
        assert not power.meets_budget(HardwareBudget(max_power_mw=1_000.0))

        latency = ResourceReport(total_latency_cycles=200)
        assert not latency.meets_budget(HardwareBudget(max_latency_cycles=100))

        ffs = ResourceReport(total_ffs=600_000)
        assert not ffs.meets_budget(HardwareBudget(max_ffs=500_000))

        dsp = ResourceReport(total_dsp=300)
        assert not dsp.meets_budget(HardwareBudget(max_dsp=256))

    def test_resource_report_summary_is_human_readable(self) -> None:
        from sc_neurocore.core.types import ResourceReport

        report = ResourceReport(
            total_luts=1_000,
            total_ffs=2_000,
            total_dsp=4,
            total_bram_kb=12.5,
            total_power_mw=3.25,
            total_latency_cycles=128,
            mean_accuracy=0.9876,
        )
        text = report.summary()
        assert "LUTs: 1000" in text
        assert "Accuracy: 0.9876" in text

    def test_layer_spec_deterministic_luts_and_power(self) -> None:
        # The deterministic mode takes the dedicated MAC-count cost paths in
        # estimate_luts and estimate_power_mw rather than the stochastic ones.
        from sc_neurocore.core.types import ComputeMode, LayerSpec

        ls = LayerSpec(layer_id="L0", neurons=8, mac_count=20, mode=ComputeMode.DETERMINISTIC)
        assert ls.estimate_luts() == 20 * 120
        assert ls.estimate_power_mw() == 20 * 0.5
