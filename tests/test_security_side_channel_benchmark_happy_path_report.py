# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (happy_path_report) from former test_security_side_channel_benchmark.py

from __future__ import annotations

from tests.security_side_channel_benchmark_support import *  # noqa: F403

def test_side_channel_benchmark_reports_protected_reduction_and_overhead() -> None:
    report = run_side_channel_leakage_benchmark(
        probabilities=(0.25, 0.5),
        labels=(0, 1),
        protected_config=ThermalSCEncodingConfig(
            bitstream_length=16,
            seed=3,
            dummy_streams_per_record=1,
            max_dummy_overhead_ratio=1.0,
        ),
    )

    assert report.schema_version == SIDE_CHANNEL_BENCHMARK_SCHEMA_VERSION
    assert report.evidence_boundary == "analytic_simulation_only"
    assert report.threat_model == "class_activity_correlation_proxy"
    assert report.baseline.class_activity_proxy.max_class_mean_gap > (
        report.protected.class_activity_proxy.max_class_mean_gap
    )
    assert report.max_class_mean_gap_reduction > 0.0
    assert report.protected.dummy_stream_overhead_ratio == 1.0
    assert report.deploy_manifest.schema_version == (SIDE_CHANNEL_DEPLOY_MANIFEST_SCHEMA_VERSION)
    assert report.deploy_manifest.security_parameters == {
        "bitstream_length": 16,
        "dummy_streams_per_record": 1,
        "max_dummy_overhead_ratio": 1.0,
        "rotation_stride": 1,
        "seed": 3,
    }
    assert report.deploy_manifest.overhead_measurements == {
        "dummy_stream_overhead_ratio": 1.0,
        "protected_bitstream_count": 2,
        "total_dummy_streams_inserted": 2,
    }
    assert report.boundary_notes == (
        "no physical power measurement",
        "no physical thermal measurement",
        "no DPA-resistance claim",
        "no silicon-security claim",
    )


def test_side_channel_benchmark_report_writes_canonical_json(tmp_path) -> None:
    output = tmp_path / "side_channel_benchmark.json"
    report = write_side_channel_benchmark_report(
        output,
        probabilities=(0.25, 0.5),
        labels=(10, 20),
        protected_config=ThermalSCEncodingConfig(bitstream_length=16, seed=5),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == SIDE_CHANNEL_BENCHMARK_SCHEMA_VERSION
    assert payload["evidence_boundary"] == "analytic_simulation_only"
    assert payload["report"]["max_class_mean_gap_reduction"] == pytest.approx(
        report.max_class_mean_gap_reduction
    )
    assert payload["report"]["protected"]["dummy_stream_overhead_ratio"] == 0.0
    assert payload["deploy_manifest"]["schema_version"] == (
        SIDE_CHANNEL_DEPLOY_MANIFEST_SCHEMA_VERSION
    )
    assert payload["deploy_manifest"]["benchmark_artifact"]["path"] == str(output)
    assert payload["deploy_manifest"]["evidence_class"] == "analytic_simulation"
    assert payload["deploy_manifest"]["security_parameters"]["bitstream_length"] == 16
    assert payload["deploy_manifest"]["overhead_measurements"]["dummy_stream_overhead_ratio"] == 0.0
    assert payload["report"]["records"][0]["label"] == 10
    assert "measured_power" not in json.dumps(payload)
    assert "measured_thermal" not in json.dumps(payload)
