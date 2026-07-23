# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio evidence bundle rejection contracts

"""Fail-closed validation for invalid action evidence and artifact state."""

from __future__ import annotations

from tests.studio_evidence_bundle_support import *  # noqa: F403

@pytest.mark.parametrize(
    ("payload_factory", "error_match"),
    [
        (lambda _job_id: b"\xff", "must be JSON"),
        (lambda _job_id: "[]", "JSON object"),
        (
            lambda job_id: json.dumps(_action_evidence_payload(job_id) | {"job_id": "sj_other"}),
            "job ID",
        ),
        (
            lambda job_id: json.dumps(_action_evidence_payload(job_id) | {"action_kind": ""}),
            "action kind",
        ),
        (
            lambda job_id: json.dumps(
                _action_evidence_payload(job_id) | {"evidence_classification": None}
            ),
            "unsupported classification",
        ),
        (
            lambda job_id: json.dumps(
                _action_evidence_payload(job_id) | {"evidence_classification": "unknown"}
            ),
            "unsupported classification",
        ),
        (
            lambda job_id: json.dumps(_action_evidence_payload(job_id) | {"status": None}),
            "unsupported status",
        ),
        (
            lambda job_id: json.dumps(_action_evidence_payload(job_id) | {"status": "running"}),
            "unsupported status",
        ),
        (
            lambda job_id: json.dumps(_action_evidence_payload(job_id) | {"payload_sha256": "bad"}),
            "payload SHA-256",
        ),
        (
            lambda job_id: json.dumps(_action_evidence_payload(job_id) | {"replay_route": None}),
            "replay route",
        ),
        (
            lambda job_id: json.dumps(_action_evidence_payload(job_id) | {"artifacts": []}),
            "artifact metadata",
        ),
        (
            lambda job_id: json.dumps(_action_evidence_payload(job_id) | {"artifacts": ["bad"]}),
            "invalid artifact metadata",
        ),
        (
            lambda job_id: json.dumps(
                _action_evidence_payload(job_id)
                | {
                    "artifacts": [
                        {
                            "relative_path": 1,
                            "sha256": "5" * 64,
                            "size_bytes": 1,
                        }
                    ]
                }
            ),
            "invalid artifact metadata",
        ),
        (
            lambda job_id: json.dumps(
                _action_evidence_payload(job_id)
                | {"artifacts": [{"relative_path": "../escape.json"}]}
            ),
            "bundle-safe",
        ),
        (
            lambda job_id: json.dumps(
                _action_evidence_payload(job_id)
                | {
                    "artifacts": [
                        {
                            "relative_path": "compiler/result.json",
                            "sha256": "bad",
                            "size_bytes": 1,
                        }
                    ]
                }
            ),
            "invalid artifact metadata",
        ),
    ],
)
def test_write_studio_evidence_bundle_rejects_invalid_action_evidence(
    tmp_path: Path,
    payload_factory: Callable[[str], bytes | str],
    error_match: str,
) -> None:
    """Evidence bundle writer fails closed on malformed worker evidence."""

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"compiler"}),
        default_timeout_seconds=1.0,
    )

    def source_task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("compiler/evidence.json", payload_factory(context.job_id))
        return {}

    source_record = manager.submit(
        kind="compiler",
        owner="studio-compiler",
        request_id=None,
        task=source_task,
    )
    completed_source = manager.wait(source_record.job_id, timeout_seconds=2.0)
    bundle_context = StudioJobContext(
        job_id="sj_evidence",
        work_dir=tmp_path / "evidence",
        cancel_event=threading.Event(),
        max_artifact_bytes=1024 * 1024,
    )

    with pytest.raises(ValueError, match=error_match):
        write_studio_evidence_bundle(
            bundle_context,
            job_records=(completed_source,),
            artifact_reader=manager.read_artifact,
        )

def test_write_studio_evidence_bundle_rejects_invalid_json_and_artifact_state(
    tmp_path: Path,
) -> None:
    """Evidence bundle writer fails closed on unsafe replay and artifact inputs."""

    context = StudioJobContext(
        job_id="sj_evidence",
        work_dir=tmp_path / "evidence",
        cancel_event=threading.Event(),
        max_artifact_bytes=1024 * 1024,
    )
    with pytest.raises(ValueError, match="command replay"):
        write_studio_evidence_bundle(
            context,
            command_replay={"bad": math.nan},
        )
    with pytest.raises(ValueError, match="command replay"):
        write_studio_evidence_bundle(
            context,
            command_replay={"bad": object()},
        )
    with pytest.raises(ValueError, match="project payload"):
        write_studio_evidence_bundle(
            context,
            project_payload=cast(dict[str, object], {1: "bad"}),
        )
    with pytest.raises(ValueError, match="project name"):
        write_studio_evidence_bundle(
            context,
            project_payload={"saved_at": 1.0, "state": {}, "version": "0.3.0"},
        )
    with pytest.raises(ValueError, match="project version"):
        write_studio_evidence_bundle(
            context,
            project_payload={"name": "demo", "saved_at": 1.0, "state": {}},
        )
    with pytest.raises(ValueError, match="project payload"):
        write_studio_evidence_bundle(
            context,
            project_payload={
                "name": "demo",
                "saved_at": math.nan,
                "state": {},
                "version": "0.3.0",
            },
        )
    with pytest.raises(ValueError, match="state object"):
        write_studio_evidence_bundle(
            context,
            project_payload={"name": "demo", "saved_at": 1.0, "state": [], "version": "0.3.0"},
        )
    with pytest.raises(ValueError, match="saved timestamp"):
        write_studio_evidence_bundle(
            context,
            project_payload={
                "name": "demo",
                "saved_at": True,
                "state": {},
                "version": "0.3.0",
            },
        )
    with pytest.raises(ValueError, match="Studio simulation payload requires run metadata"):
        write_studio_evidence_bundle(
            context,
            simulation_payloads=({"time": []},),
        )
    invalid_simulation = _simulation_payload()
    invalid_simulation["run_metadata"] = {"schema_version": "legacy"}
    with pytest.raises(ValueError, match="unsupported run metadata"):
        write_studio_evidence_bundle(
            context,
            simulation_payloads=(invalid_simulation,),
        )
    with pytest.raises(ValueError, match="Studio analysis payload requires analysis metadata"):
        write_studio_evidence_bundle(
            context,
            analysis_payloads=({"rates": []},),
        )
    invalid_analysis = _analysis_payload()
    invalid_analysis["analysis_metadata"] = {"schema_version": "legacy"}
    with pytest.raises(ValueError, match="unsupported analysis metadata"):
        write_studio_evidence_bundle(
            context,
            analysis_payloads=(invalid_analysis,),
        )
    invalid_simulation_classification = _simulation_payload()
    invalid_simulation_classification["run_metadata"] = {
        "evidence_classification": "analysis",
        "schema_version": "studio.simulation-run.v1",
        "status": "completed",
    }
    with pytest.raises(ValueError, match="classified as simulation evidence"):
        write_studio_evidence_bundle(
            context,
            simulation_payloads=(invalid_simulation_classification,),
        )
    invalid_analysis_classification = _analysis_payload()
    invalid_analysis_classification["analysis_metadata"] = {
        "evidence_classification": "simulation",
        "schema_version": "studio.analysis-result.v1",
        "status": "completed",
    }
    with pytest.raises(ValueError, match="classified as analysis evidence"):
        write_studio_evidence_bundle(
            context,
            analysis_payloads=(invalid_analysis_classification,),
        )
    invalid_simulation_status = _simulation_payload()
    cast(dict[str, object], invalid_simulation_status["run_metadata"])["status"] = "failed"
    with pytest.raises(ValueError, match="completed evidence status"):
        write_studio_evidence_bundle(
            context,
            simulation_payloads=(invalid_simulation_status,),
        )
    invalid_analysis_status = _analysis_payload()
    cast(dict[str, object], invalid_analysis_status["analysis_metadata"])["status"] = "failed"
    with pytest.raises(ValueError, match="completed evidence status"):
        write_studio_evidence_bundle(
            context,
            analysis_payloads=(invalid_analysis_status,),
        )
    with pytest.raises(ValueError, match="default-flow run payload has unsupported schema"):
        write_studio_evidence_bundle(
            context,
            default_flow_runs=({"schema_version": "legacy"},),
        )
    invalid_default_flow_run = _default_flow_run_payload()
    invalid_default_flow_run["preset_id"] = ""
    with pytest.raises(ValueError, match="requires a preset ID"):
        write_studio_evidence_bundle(
            context,
            default_flow_runs=(invalid_default_flow_run,),
        )
    invalid_default_flow_run = _default_flow_run_payload()
    invalid_default_flow_run["evidence_classification"] = "analysis"
    with pytest.raises(ValueError, match="classified as default-flow evidence"):
        write_studio_evidence_bundle(
            context,
            default_flow_runs=(invalid_default_flow_run,),
        )
    invalid_default_flow_run = _default_flow_run_payload()
    invalid_default_flow_run["status"] = "failed"
    with pytest.raises(ValueError, match="completed evidence status"):
        write_studio_evidence_bundle(
            context,
            default_flow_runs=(invalid_default_flow_run,),
        )
    invalid_default_flow_run = _default_flow_run_payload()
    invalid_default_flow_run["flow_id"] = ""
    with pytest.raises(ValueError, match="requires a flow ID"):
        write_studio_evidence_bundle(
            context,
            default_flow_runs=(invalid_default_flow_run,),
        )
    invalid_default_flow_run = _default_flow_run_payload()
    invalid_default_flow_run["action_order"] = [""]
    with pytest.raises(ValueError, match="requires action order"):
        write_studio_evidence_bundle(
            context,
            default_flow_runs=(invalid_default_flow_run,),
        )
    invalid_default_flow_run = _default_flow_run_payload()
    invalid_default_flow_run["executed_count"] = -1
    with pytest.raises(ValueError, match="requires executed count"):
        write_studio_evidence_bundle(
            context,
            default_flow_runs=(invalid_default_flow_run,),
        )
    invalid_default_flow_run = _default_flow_run_payload()
    invalid_default_flow_run["reproducibility_manifest"] = None
    with pytest.raises(ValueError, match="requires reproducibility metadata"):
        write_studio_evidence_bundle(
            context,
            default_flow_runs=(invalid_default_flow_run,),
        )
    invalid_default_flow_run = _default_flow_run_payload()
    invalid_default_flow_run["reproducibility_manifest"] = {"hash_algorithm": "md5"}
    with pytest.raises(ValueError, match="unsupported hash algorithm"):
        write_studio_evidence_bundle(
            context,
            default_flow_runs=(invalid_default_flow_run,),
        )
    invalid_default_flow_run = _default_flow_run_payload()
    invalid_default_flow_run["reproducibility_manifest"] = {
        "hash_algorithm": "sha256",
        "inputs_fingerprint_sha256": "bad",
        "run_fingerprint_sha256": "8" * 64,
    }
    with pytest.raises(ValueError, match="requires SHA-256 fingerprints"):
        write_studio_evidence_bundle(
            context,
            default_flow_runs=(invalid_default_flow_run,),
        )
    with pytest.raises(ValueError, match="default-flow attestation payload has unsupported schema"):
        write_studio_evidence_bundle(
            context,
            default_flow_attestations=({"schema_version": "legacy"},),
        )
    invalid_attestation = _default_flow_attestation_payload()
    invalid_attestation["preset_id"] = ""
    with pytest.raises(ValueError, match="requires a preset ID"):
        write_studio_evidence_bundle(
            context,
            default_flow_attestations=(invalid_attestation,),
        )
    invalid_attestation = _default_flow_attestation_payload()
    invalid_attestation["evidence_classification"] = "analysis"
    with pytest.raises(ValueError, match="classified as default-flow evidence"):
        write_studio_evidence_bundle(
            context,
            default_flow_attestations=(invalid_attestation,),
        )
    invalid_attestation = _default_flow_attestation_payload()
    invalid_attestation["status"] = "failed"
    with pytest.raises(ValueError, match="completed evidence status"):
        write_studio_evidence_bundle(
            context,
            default_flow_attestations=(invalid_attestation,),
        )
    invalid_attestation = _default_flow_attestation_payload()
    invalid_attestation["flow_id"] = ""
    with pytest.raises(ValueError, match="requires a flow ID"):
        write_studio_evidence_bundle(
            context,
            default_flow_attestations=(invalid_attestation,),
        )
    invalid_attestation = _default_flow_attestation_payload()
    invalid_attestation["plan_fingerprint_sha256"] = "bad"
    with pytest.raises(ValueError, match="requires SHA-256 fingerprints"):
        write_studio_evidence_bundle(
            context,
            default_flow_attestations=(invalid_attestation,),
        )
    mismatched_attestation = _default_flow_attestation_payload()
    mismatched_attestation["run_fingerprint_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="does not match supplied run"):
        write_studio_evidence_bundle(
            context,
            default_flow_runs=(_default_flow_run_payload(),),
            default_flow_attestations=(mismatched_attestation,),
        )

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"compiler"}),
        default_timeout_seconds=1.0,
    )

    def source_task(job_context: StudioJobContext) -> dict[str, object]:
        job_context.write_artifact("compiler/result.json", "{}")
        return {}

    source_record = manager.submit(
        kind="compiler",
        owner="studio-compiler",
        request_id=None,
        task=source_task,
    )
    completed_source = manager.wait(source_record.job_id, timeout_seconds=2.0)
    with pytest.raises(ValueError, match="artifact reader"):
        write_studio_evidence_bundle(context, job_records=(completed_source,))
    unsafe_record = StudioJobRecord(
        job_id="sj_unsafe",
        kind="compiler",
        owner="operator",
        request_id=None,
        status="completed",
        execution_model="thread",
        created_at_utc="2026-06-20T00:00:00Z",
        artifacts=(
            StudioJobArtifact(
                relative_path="../escape.json",
                size_bytes=2,
                sha256="0" * 64,
            ),
        ),
    )
    with pytest.raises(ValueError, match="bundle-safe"):
        write_studio_evidence_bundle(
            context,
            job_records=(unsafe_record,),
            artifact_reader=manager.read_artifact,
        )

    def corrupt_evidence_task(job_context: StudioJobContext) -> dict[str, object]:
        job_context.write_artifact("compiler/evidence.json", '{"schema_version": "legacy"}')
        return {}

    corrupt_record = manager.submit(
        kind="compiler",
        owner="studio-compiler",
        request_id=None,
        task=corrupt_evidence_task,
    )
    completed_corrupt = manager.wait(corrupt_record.job_id, timeout_seconds=2.0)
    with pytest.raises(ValueError, match="unsupported schema"):
        write_studio_evidence_bundle(
            context,
            job_records=(completed_corrupt,),
            artifact_reader=manager.read_artifact,
        )
