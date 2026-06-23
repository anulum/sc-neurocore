# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import json

import pytest
from sc_neurocore.security import side_channel_benchmark as benchmark_mod

from sc_neurocore.security.side_channel_benchmark import (
    SIDE_CHANNEL_BENCHMARK_SCHEMA_VERSION,
    SIDE_CHANNEL_DEPLOY_MANIFEST_SCHEMA_VERSION,
    SideChannelBenchmarkArm,
    SideChannelBenchmarkError,
    SideChannelBenchmarkRecord,
    SideChannelDeployManifest,
    SideChannelBenchmarkReport,
    _correlated_activity_fixture_stream,
    _arm_payload,
    _class_proxy_payload,
    _deploy_manifest_payload,
    _report_payload,
    _with_artifact_path,
    run_side_channel_leakage_benchmark,
    write_side_channel_benchmark_report,
)
from sc_neurocore.security.side_channel_metrics import (
    SideChannelMetricError,
    compute_class_activity_proxy,
    compute_switching_activity,
)
from sc_neurocore.security.thermal_sc_encoding import ThermalSCEncodingConfig


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


@pytest.mark.parametrize(
    ("probabilities", "labels"),
    [
        ((), ()),
        ("invalid", (0, 1)),
        ((0.25,), (0,)),
        ((0.25, 0.5), "invalid"),
        ((0.25,), (0, 1)),
        ((0.25, 0.5), (0,)),
        ((0.25, float("nan")), (0, 1)),
        ((0.25, 0.5), (0, float("nan"))),
        ((0.25, 0.5), (0, True)),
        ((0.25, True), (0, 1)),
    ],
)
def test_side_channel_benchmark_rejects_invalid_sample_contracts(
    probabilities: tuple[float, ...],
    labels: tuple[int, ...],
) -> None:
    with pytest.raises(SideChannelBenchmarkError):
        run_side_channel_leakage_benchmark(
            probabilities=probabilities,
            labels=labels,
            protected_config=ThermalSCEncodingConfig(bitstream_length=16),
        )


def test_side_channel_benchmark_maps_thermal_encoder_contract_errors() -> None:
    with pytest.raises(SideChannelBenchmarkError, match="dummy stream insertion exceeds"):
        run_side_channel_leakage_benchmark(
            probabilities=(0.25, 0.5),
            labels=(0, 1),
            protected_config=ThermalSCEncodingConfig(
                bitstream_length=16,
                dummy_streams_per_record=2,
                max_dummy_overhead_ratio=1.0,
            ),
        )


def test_side_channel_benchmark_rejects_invalid_protected_config() -> None:
    with pytest.raises(SideChannelBenchmarkError, match="protected_config"):
        run_side_channel_leakage_benchmark(
            probabilities=(0.25, 0.5),
            labels=(0, 1),
            protected_config="bad",  # type: ignore[arg-type]
        )


def test_side_channel_benchmark_rejects_mismatched_encoder_batch_length(monkeypatch) -> None:
    class _DummySummary:
        dummy_stream_overhead_ratio = 0.0
        class_activity_proxy = compute_class_activity_proxy((((0, 1),), ((1, 0),)), (0, 1))

    class _DummyRecord:
        realised_probability = 0.5
        dummy_streams_inserted = 0

    class _DummyBatch:
        summary = _DummySummary()
        records = (_DummyRecord(),)

    monkeypatch.setattr(
        benchmark_mod, "encode_activity_balanced_probabilities", lambda *a, **k: _DummyBatch()
    )

    with pytest.raises(SideChannelBenchmarkError, match="output length"):
        run_side_channel_leakage_benchmark(
            probabilities=(0.25, 0.5),
            labels=(0, 1),
            protected_config=ThermalSCEncodingConfig(bitstream_length=16, seed=3),
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": ""}, "name"),
        ({"class_activity_proxy": "bad"}, "class_activity_proxy"),
        ({"dummy_stream_overhead_ratio": True}, "dummy_stream_overhead_ratio"),
        ({"dummy_stream_overhead_ratio": -0.1}, "dummy_stream_overhead_ratio"),
        ({"dummy_stream_overhead_ratio": float("nan")}, "dummy_stream_overhead_ratio"),
        ({"bitstream_count": -1}, "bitstream_count"),
    ],
)
def test_side_channel_benchmark_arm_rejects_invalid_contracts(kwargs, match) -> None:
    proxy = compute_class_activity_proxy((((0, 1),), ((1, 0),)), (0, 1))
    values = {
        "name": "baseline",
        "class_activity_proxy": proxy,
        "dummy_stream_overhead_ratio": 0.0,
        "bitstream_count": 2,
    }
    values.update(kwargs)
    with pytest.raises(SideChannelBenchmarkError, match=match):
        SideChannelBenchmarkArm(**values)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"label": True}, "label"),
        ({"label": float("nan")}, "label"),
        ({"probability": True}, "probability"),
        ({"probability": 1.1}, "probability"),
        ({"protected_realised_probability": -0.1}, "protected_realised_probability"),
        ({"protected_dummy_streams_inserted": -1}, "protected_dummy_streams_inserted"),
    ],
)
def test_side_channel_benchmark_record_rejects_invalid_contracts(kwargs, match) -> None:
    values = {
        "label": 0,
        "probability": 0.5,
        "protected_realised_probability": 0.5,
        "protected_dummy_streams_inserted": 0,
    }
    values.update(kwargs)
    with pytest.raises(SideChannelBenchmarkError, match=match):
        SideChannelBenchmarkRecord(**values)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"schema_version": ""}, "schema_version"),
        ({"evidence_class": ""}, "evidence_class"),
        ({"benchmark_artifact": "bad"}, "benchmark_artifact"),
        ({"security_parameters": "bad"}, "security_parameters"),
        ({"overhead_measurements": "bad"}, "overhead_measurements"),
        ({"boundary_notes": ()}, "boundary_notes"),
        ({"boundary_notes": ("",)}, "boundary_notes"),
    ],
)
def test_side_channel_deploy_manifest_rejects_invalid_contracts(kwargs, match) -> None:
    values = {
        "schema_version": SIDE_CHANNEL_DEPLOY_MANIFEST_SCHEMA_VERSION,
        "evidence_class": "analytic_simulation",
        "benchmark_artifact": {"path": "x.json"},
        "security_parameters": {"bitstream_length": 16},
        "overhead_measurements": {"dummy_stream_overhead_ratio": 0.0},
        "boundary_notes": ("note",),
    }
    values.update(kwargs)
    with pytest.raises(SideChannelBenchmarkError, match=match):
        SideChannelDeployManifest(**values)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"schema_version": ""}, "schema_version"),
        ({"evidence_boundary": ""}, "evidence_boundary"),
        ({"threat_model": ""}, "threat_model"),
        ({"baseline": "bad"}, "baseline"),
        ({"protected": "bad"}, "protected"),
        ({"max_class_mean_gap_reduction": True}, "max_class_mean_gap_reduction"),
        ({"max_class_mean_gap_reduction": float("nan")}, "max_class_mean_gap_reduction"),
        ({"deploy_manifest": "bad"}, "deploy_manifest"),
        ({"boundary_notes": ()}, "boundary_notes"),
        ({"boundary_notes": ("",)}, "boundary_notes"),
        ({"records": ["bad"]}, "records"),
        ({"records": ("bad",)}, "records"),
    ],
)
def test_side_channel_benchmark_report_rejects_invalid_contracts(kwargs, match) -> None:
    report = run_side_channel_leakage_benchmark(
        probabilities=(0.25, 0.5),
        labels=(0, 1),
        protected_config=ThermalSCEncodingConfig(bitstream_length=16, seed=3),
    )
    values = {
        "schema_version": report.schema_version,
        "evidence_boundary": report.evidence_boundary,
        "threat_model": report.threat_model,
        "baseline": report.baseline,
        "protected": report.protected,
        "max_class_mean_gap_reduction": report.max_class_mean_gap_reduction,
        "deploy_manifest": report.deploy_manifest,
        "boundary_notes": report.boundary_notes,
        "records": report.records,
    }
    values.update(kwargs)
    with pytest.raises(SideChannelBenchmarkError, match=match):
        SideChannelBenchmarkReport(**values)


def test_with_artifact_path_rejects_invalid_contracts(tmp_path) -> None:
    report = run_side_channel_leakage_benchmark(
        probabilities=(0.25, 0.5),
        labels=(0, 1),
        protected_config=ThermalSCEncodingConfig(bitstream_length=16, seed=3),
    )
    with pytest.raises(SideChannelBenchmarkError, match="report"):
        _with_artifact_path("bad", tmp_path / "a.json")  # type: ignore[arg-type]
    with pytest.raises(SideChannelBenchmarkError, match="path"):
        _with_artifact_path(report, "bad")  # type: ignore[arg-type]


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
