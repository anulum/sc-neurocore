# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestResilienceReportContracts from former test_fault_injection_module.py

"""Focused suite: TestResilienceReportContracts from former test_fault_injection_module.py."""

from __future__ import annotations

from typing import Any
import pytest
from sc_neurocore.fault_injection.fault_injection import (
    ResilienceReport,
)


class TestResilienceReportContracts:
    """Checks on resilience reports."""

    def test_summary_includes_core_metrics(self) -> None:
        """The summary names the fault and trials."""
        report = ResilienceReport(
            fault_model="bit_flip",
            ber=1e-3,
            bitstream_length=128,
            num_trials=10,
            mean_error=0.01,
            std_error=0.005,
            max_error=0.03,
            p95_error=0.02,
            p99_error=0.025,
            mean_bits_flipped=1.5,
            wall_time_ms=2.5,
        )
        text = report.summary()
        assert "Fault: bit_flip" in text
        assert "Trials=10" in text

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("ber", 1.5, "ber"),
            ("bitstream_length", 0, "bitstream_length"),
            ("num_trials", 0, "num_trials"),
            ("mean_error", -0.1, "mean_error"),
            ("p95_error", 0.001, "p95_error"),
            ("p99_error", 0.019, "p99_error"),
            ("max_error", 0.024, "max_error"),
            ("mean_bits_flipped", 129.0, "mean_bits_flipped"),
            ("wall_time_ms", -1.0, "wall_time_ms"),
        ],
    )
    def test_rejects_invalid_contracts(self, field: str, value: object, match: str) -> None:
        """Each invalid report field is rejected."""
        values: dict[str, Any] = {
            "fault_model": "bit_flip",
            "ber": 1e-3,
            "bitstream_length": 128,
            "num_trials": 10,
            "mean_error": 0.01,
            "std_error": 0.005,
            "max_error": 0.03,
            "p95_error": 0.02,
            "p99_error": 0.025,
            "mean_bits_flipped": 1.5,
            "wall_time_ms": 2.5,
        }
        values[field] = value
        with pytest.raises(ValueError, match=match):
            ResilienceReport(**values)

    def test_rejects_unknown_fault_model_value(self) -> None:
        """An unknown fault model is rejected."""
        with pytest.raises(ValueError, match="known FaultModel"):
            ResilienceReport(
                fault_model="unknown_fault",
                ber=1e-3,
                bitstream_length=128,
                num_trials=10,
                mean_error=0.01,
                std_error=0.005,
                max_error=0.03,
                p95_error=0.02,
                p99_error=0.025,
                mean_bits_flipped=1.5,
                wall_time_ms=2.5,
            )
