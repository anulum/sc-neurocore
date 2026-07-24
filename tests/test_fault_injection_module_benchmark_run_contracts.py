# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBenchmarkRunContracts from former test_fault_injection_module.py

"""Focused suite: TestBenchmarkRunContracts from former test_fault_injection_module.py."""

from __future__ import annotations

from tests.fault_injection_module_support import *  # noqa: F403


class TestBenchmarkRunContracts:
    def test_run_returns_report_with_expected_fault_model(self):
        bench = ResilienceBenchmark(seed=2)
        report = bench.run(
            fault_model=FaultModel.BIT_FLIP, ber=1e-3, bitstream_length=32, num_trials=5
        )
        assert report.fault_model == FaultModel.BIT_FLIP.value
        assert report.num_trials == 5

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"fault_model": "bit_flip"}, "fault_model"),
            ({"ber": 1.2}, "ber"),
            ({"bitstream_length": 0}, "bitstream_length"),
            ({"probability": -0.1}, "probability"),
            ({"num_trials": 0}, "num_trials"),
        ],
    )
    def test_run_rejects_invalid_inputs(self, kwargs, match):
        bench = ResilienceBenchmark(seed=2)
        values = {
            "fault_model": FaultModel.BIT_FLIP,
            "ber": 1e-3,
            "bitstream_length": 32,
            "probability": 0.5,
            "num_trials": 5,
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            bench.run(**values)  # type: ignore[arg-type]
