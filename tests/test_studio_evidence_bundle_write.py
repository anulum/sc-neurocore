# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio evidence bundle write contracts

"""Happy-path and classification contracts for write_studio_evidence_bundle."""

from __future__ import annotations

from tests.studio_evidence_bundle_support import *  # noqa: F403


def test_write_studio_evidence_bundle_copies_project_job_audit_and_replay(
    tmp_path: Path,
) -> None:
    """Evidence bundle writer preserves project, job, audit, and artifact data."""

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"compiler"}),
        default_timeout_seconds=1.0,
    )

    def source_task(context: StudioJobContext) -> dict[str, object]:
        result: dict[str, object] = {"compiled": True}
        result_artifact = context.write_artifact("compiler/result.json", json.dumps(result))
        write_studio_action_evidence_manifest(
            context,
            action_kind="studio.compile",
            result=result,
            result_artifact=result_artifact,
            evidence_artifact_path="compiler/evidence.json",
            evidence_classification="compile",
            replay_route="POST /api/compile",
        )
        return result

    source_record = manager.submit(
        kind="compiler",
        owner="studio-compiler",
        request_id="req-1",
        task=source_task,
    )
    completed_source = manager.wait(source_record.job_id, timeout_seconds=2.0)
    bundle_context = StudioJobContext(
        job_id="sj_evidence",
        work_dir=tmp_path / "evidence",
        cancel_event=threading.Event(),
        max_artifact_bytes=1024 * 1024,
    )

    result = write_studio_evidence_bundle(
        bundle_context,
        project_payload=_project_payload(),
        simulation_payloads=(_simulation_payload(),),
        analysis_payloads=(_analysis_payload(),),
        default_flow_runs=(_default_flow_run_payload(),),
        default_flow_attestations=(_default_flow_attestation_payload(),),
        job_records=(completed_source,),
        artifact_reader=manager.read_artifact,
        audit_export={"schema_version": "studio.audit.export.v1", "events": []},
        command_replay={"method": "POST", "path": "/api/compile"},
        clock=lambda: datetime(2026, 6, 20, tzinfo=UTC),
    )
    payload = result.to_public_dict()

    assert payload["schema_version"] == STUDIO_EVIDENCE_BUNDLE_SCHEMA_VERSION
    assert payload["bundle_id"] == "seb_sj_evidence"
    assert "evidence/manifest.json" in result.artifact_paths
    assert "evidence/project.json" in result.artifact_paths
    assert "evidence/simulations/000.json" in result.artifact_paths
    assert "evidence/analyses/000.json" in result.artifact_paths
    assert "evidence/default-flows/runs/000.json" in result.artifact_paths
    assert "evidence/default-flows/attestations/000.json" in result.artifact_paths
    assert f"evidence/jobs/{source_record.job_id}/record.json" in result.artifact_paths
    assert (
        f"evidence/jobs/{source_record.job_id}/artifacts/compiler/result.json"
        in result.artifact_paths
    )
    assert (
        f"evidence/jobs/{source_record.job_id}/artifacts/compiler/evidence.json"
        in result.artifact_paths
    )
    assert (tmp_path / "evidence" / "evidence" / "command-replay.json").is_file()
    assert "compiler/result.json" in json.dumps(payload)
    assert "simulation_result" in json.dumps(payload)
    assert "analysis_result" in json.dumps(payload)
    assert "default_flow_run" in json.dumps(payload)
    assert "default_flow_attestation" in json.dumps(payload)
    assert "action_evidence" in json.dumps(payload)
    assert result.manifest["summary"] == payload["summary"]
    summary = cast(dict[str, object], payload["summary"])
    entry_type_counts = cast(dict[str, int], summary["entry_type_counts"])
    evidence_classification_counts = cast(
        dict[str, int],
        summary["evidence_classification_counts"],
    )
    source_job_kind_counts = cast(dict[str, int], summary["source_job_kind_counts"])
    source_job_owner_counts = cast(dict[str, int], summary["source_job_owner_counts"])
    assert summary["artifact_path_count"] == len(result.artifact_paths)
    assert summary["entry_count"] == 10
    assert entry_type_counts["action_evidence"] == 1
    assert entry_type_counts["default_flow_attestation"] == 1
    assert entry_type_counts["default_flow_run"] == 1
    assert entry_type_counts["analysis_result"] == 1
    assert entry_type_counts["simulation_result"] == 1
    assert evidence_classification_counts["analysis"] == 1
    assert evidence_classification_counts["compile"] == 1
    assert evidence_classification_counts["default_flow"] == 2
    assert evidence_classification_counts["project_workspace"] == 1
    assert evidence_classification_counts["simulation"] == 1
    assert summary["known_evidence_classifications"] == [
        "analysis",
        "compile",
        "cosim_parity",
        "default_flow",
        "local_regression",
        "project_workspace",
        "release_benchmark",
        "simulation",
        "synthesis",
        "training",
    ]
    assert summary["known_terminal_statuses"] == [
        "cancelled",
        "completed",
        "failed",
        "timed_out",
    ]
    assert source_job_kind_counts["compiler"] == 1
    assert source_job_owner_counts["studio-compiler"] == 1
    manifest = cast(dict[str, object], payload["manifest"])
    manifest_entries = cast(list[dict[str, object]], manifest["entries"])
    project_entry = next(entry for entry in manifest_entries if entry["type"] == "project")
    assert project_entry["evidence_classification"] == "project_workspace"


def test_write_studio_evidence_bundle_classifies_suffixed_action_evidence(
    tmp_path: Path,
) -> None:
    """Evidence bundles classify suffixed worker evidence artifacts."""

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=1.0,
    )

    def source_task(context: StudioJobContext) -> dict[str, object]:
        result: dict[str, object] = {"supported": True}
        result_artifact = context.write_artifact(
            "synthesis/multi-target-result.json",
            json.dumps(result),
        )
        write_studio_action_evidence_manifest(
            context,
            action_kind="studio.synthesis.multi_target",
            result=result,
            result_artifact=result_artifact,
            evidence_artifact_path="synthesis/multi-target-evidence.json",
            evidence_classification="synthesis",
            replay_route="POST /api/synth/multi-target",
        )
        return result

    source_record = manager.submit(
        kind="synthesis",
        owner="studio-synthesis",
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

    result = write_studio_evidence_bundle(
        bundle_context,
        job_records=(completed_source,),
        artifact_reader=manager.read_artifact,
    )

    encoded_manifest = json.dumps(result.manifest)
    assert "synthesis/multi-target-evidence.json" in encoded_manifest
    assert "studio.synthesis.multi_target" in encoded_manifest
    assert "action_evidence" in encoded_manifest
