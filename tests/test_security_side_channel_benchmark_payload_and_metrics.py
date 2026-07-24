# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (payload_and_metrics) from former test_security_side_channel_benchmark.py

from __future__ import annotations

from tests.security_side_channel_benchmark_support import *  # noqa: F403

def test_deploy_manifest_payload_rejects_invalid_manifest() -> None:
    with pytest.raises(SideChannelBenchmarkError, match="manifest"):
        _deploy_manifest_payload("bad")  # type: ignore[arg-type]


def test_arm_payload_rejects_invalid_arm() -> None:
    with pytest.raises(SideChannelBenchmarkError, match="arm"):
        _arm_payload("bad")  # type: ignore[arg-type]


def test_class_proxy_payload_rejects_invalid_proxy() -> None:
    with pytest.raises(SideChannelBenchmarkError, match="proxy"):
        _class_proxy_payload("bad")  # type: ignore[arg-type]


def test_report_payload_rejects_invalid_report() -> None:
    with pytest.raises(SideChannelBenchmarkError, match="report"):
        _report_payload("bad")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("probability", "bitstream_length", "match"),
    [
        (True, 16, "probability"),
        (1.1, 16, "probability"),
        (float("nan"), 16, "probability"),
        (0.5, 0, "bitstream_length"),
    ],
)
def test_correlated_activity_fixture_stream_rejects_invalid_contracts(
    probability, bitstream_length, match
) -> None:
    with pytest.raises(SideChannelBenchmarkError, match=match):
        _correlated_activity_fixture_stream(probability, bitstream_length)


def test_class_activity_proxy_rejects_non_sequence_sample() -> None:
    # Each per-class entry must be a bitstream matrix; a scalar sample cannot be
    # normalised into rows of cycles.
    with pytest.raises(SideChannelMetricError, match="non-empty bitstream matrix"):
        compute_class_activity_proxy([42], [0])


def test_switching_activity_rejects_ragged_matrix() -> None:
    # Rows of differing cycle counts cannot form a rectangular bitstream matrix.
    with pytest.raises(SideChannelMetricError, match="must be rectangular"):
        compute_switching_activity([[1, 0, 1], [1, 0]])
