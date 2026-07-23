# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBenchmarkSweepContracts from former test_fault_injection_module.py

"""Focused suite: TestBenchmarkSweepContracts from former test_fault_injection_module.py."""

from __future__ import annotations

from tests.fault_injection_module_support import *  # noqa: F403

class TestBenchmarkSweepContracts:
    def test_sweep_returns_reports_for_each_ber(self):
        bench = ResilienceBenchmark(seed=2)
        reports = bench.sweep_ber(
            fault_model=FaultModel.BIT_FLIP,
            ber_range=[1e-4, 1e-3, 1e-2],
            bitstream_length=32,
            num_trials=5,
        )
        assert len(reports) == 3
        assert [r.ber for r in reports] == [1e-4, 1e-3, 1e-2]

    @pytest.mark.parametrize(
        ("ber_range", "match"),
        [
            ([], "non-empty"),
            ([1e-3, 1e-4], "monotonically"),
            ([1e-3, 1.2], "entries"),
        ],
    )
    def test_sweep_rejects_invalid_ber_range(self, ber_range, match):
        bench = ResilienceBenchmark(seed=2)
        with pytest.raises(ValueError, match=match):
            bench.sweep_ber(
                fault_model=FaultModel.BIT_FLIP,
                ber_range=ber_range,  # type: ignore[arg-type]
                bitstream_length=32,
                num_trials=5,
            )
