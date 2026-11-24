# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (write_path_rejects) from former test_security_side_channel_benchmark.py

"""Output paths the side-channel benchmark report writer refuses."""

from __future__ import annotations

from typing import Any, cast
from pathlib import Path
import pytest
from sc_neurocore.security.side_channel_benchmark import (
    SideChannelBenchmarkError,
    write_side_channel_benchmark_report,
)
from sc_neurocore.security.thermal_sc_encoding import (
    ThermalSCEncodingConfig,
)


@pytest.mark.parametrize("output_path", ["", "."])
def test_write_side_channel_benchmark_report_rejects_invalid_output_path(output_path: str) -> None:
    """Write side channel benchmark report rejects invalid output path."""
    with pytest.raises(SideChannelBenchmarkError, match="output_path"):
        write_side_channel_benchmark_report(
            output_path,
            probabilities=(0.25, 0.5),
            labels=(0, 1),
            protected_config=ThermalSCEncodingConfig(bitstream_length=16, seed=3),
        )


def test_write_side_channel_benchmark_report_rejects_non_path_output_type() -> None:
    """Write side channel benchmark report rejects non-path output type."""
    with pytest.raises(SideChannelBenchmarkError, match="output_path must be a string or Path"):
        write_side_channel_benchmark_report(
            cast(Any, 123),
            probabilities=(0.25, 0.5),
            labels=(0, 1),
            protected_config=ThermalSCEncodingConfig(bitstream_length=16, seed=3),
        )


def test_write_side_channel_benchmark_report_rejects_directory_output_path(tmp_path: Path) -> None:
    """Write side channel benchmark report rejects directory output path."""
    with pytest.raises(SideChannelBenchmarkError, match="existing directory"):
        write_side_channel_benchmark_report(
            tmp_path,
            probabilities=(0.25, 0.5),
            labels=(0, 1),
            protected_config=ThermalSCEncodingConfig(bitstream_length=16, seed=3),
        )
