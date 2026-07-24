# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (write_path_rejects) from former test_security_side_channel_benchmark.py

from __future__ import annotations

from tests.security_side_channel_benchmark_support import *  # noqa: F403

@pytest.mark.parametrize("output_path", ["", "."])
def test_write_side_channel_benchmark_report_rejects_invalid_output_path(output_path) -> None:
    with pytest.raises(SideChannelBenchmarkError, match="output_path"):
        write_side_channel_benchmark_report(
            output_path,
            probabilities=(0.25, 0.5),
            labels=(0, 1),
            protected_config=ThermalSCEncodingConfig(bitstream_length=16, seed=3),
        )


def test_write_side_channel_benchmark_report_rejects_non_path_output_type() -> None:
    with pytest.raises(SideChannelBenchmarkError, match="output_path must be a string or Path"):
        write_side_channel_benchmark_report(
            123,  # type: ignore[arg-type]
            probabilities=(0.25, 0.5),
            labels=(0, 1),
            protected_config=ThermalSCEncodingConfig(bitstream_length=16, seed=3),
        )


def test_write_side_channel_benchmark_report_rejects_directory_output_path(tmp_path) -> None:
    with pytest.raises(SideChannelBenchmarkError, match="existing directory"):
        write_side_channel_benchmark_report(
            tmp_path,
            probabilities=(0.25, 0.5),
            labels=(0, 1),
            protected_config=ThermalSCEncodingConfig(bitstream_length=16, seed=3),
        )
