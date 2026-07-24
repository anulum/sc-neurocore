# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBenchmarkRunAggregateGuards from former test_fault_injection_module.py

"""Focused suite: TestBenchmarkRunAggregateGuards from former test_fault_injection_module.py."""

from __future__ import annotations

from tests.fault_injection_module_support import *  # noqa: F403


class TestBenchmarkRunAggregateGuards:
    def test_run_outputs_finite_statistics(self):
        bench = ResilienceBenchmark(seed=5)
        report = bench.run(
            fault_model=FaultModel.BIT_FLIP,
            ber=0.01,
            bitstream_length=64,
            probability=0.4,
            num_trials=8,
        )
        assert report.mean_error >= 0.0
        assert report.max_error >= report.p99_error >= report.p95_error
        assert 0.0 <= report.mean_bits_flipped <= 64.0
