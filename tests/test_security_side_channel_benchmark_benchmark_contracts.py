# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (benchmark_contracts) from former test_security_side_channel_benchmark.py

from __future__ import annotations

from tests.security_side_channel_benchmark_support import *  # noqa: F403

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
