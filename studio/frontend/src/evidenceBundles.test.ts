// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio evidence bundle helper tests
import { describe, expect, it } from "vitest";

import type {
  StudioAuditStatus,
  StudioEvidenceBundleResponse,
  StudioJobListResponse,
  StudioJobRecord,
  StudioJobStatus,
  StudioOperatorStatus,
} from "./api/client";
import {
  adminEvidenceBundleCreatedState,
  adminEvidenceBundleFailureState,
  adminEvidenceBundleLoadingState,
  evidenceBundleArtifactDownloadPlan,
  evidenceBundleArtifactDownloadFailureState,
  evidenceBundleArtifactDownloadStartState,
  evidenceBundleArtifactUnavailableState,
  evidenceBundleDownloadSelection,
  evidenceBundleSurfaceKeys,
  latestSynthesisJobIdWithArtefact,
  scopedEvidenceBundleCreatedState,
  scopedEvidenceBundleFailureState,
  scopedEvidenceBundleLoadingState,
  selectEvidenceBundleForSurface,
} from "./evidenceBundles";

const bundle: StudioEvidenceBundleResponse = {
  artifact_paths: ["evidence/manifest.json"],
  artifacts: [],
  bundle_id: "seb_project",
  job_id: "sj_bundle",
  manifest: {},
  schema_version: "studio.evidence-bundle.v1",
  summary: {
    artifact_path_count: 1,
    entry_count: 1,
    entry_type_counts: { manifest: 1 },
    evidence_classification_counts: {},
    source_job_count: 0,
    source_job_kind_counts: {},
    source_job_owner_counts: {},
  },
};

function jobRecord(
  jobId: string,
  createdAt: string,
  kind: string,
  artefactPaths: string[],
): StudioJobRecord {
  return {
    artifacts: artefactPaths.map((relativePath) => ({
      relative_path: relativePath,
      sha256: "a".repeat(64),
      size_bytes: 128,
    })),
    created_at_utc: createdAt,
    error: null,
    execution_model: "process",
    finished_at_utc: createdAt,
    job_id: jobId,
    kind,
    owner: "studio",
    request_id: null,
    result: null,
    started_at_utc: createdAt,
    status: "completed",
  };
}

function auditStatus(overrides: Partial<StudioAuditStatus> = {}): StudioAuditStatus {
  return {
    configured: overrides.configured ?? true,
    healthy: overrides.healthy ?? true,
    last_error: overrides.last_error ?? null,
    path_configured: overrides.path_configured ?? true,
    sink_type: overrides.sink_type ?? "jsonl",
  };
}

function jobStatus(overrides: Partial<StudioJobStatus> = {}): StudioJobStatus {
  return {
    active_count: overrides.active_count ?? 1,
    allowed_kinds: overrides.allowed_kinds ?? ["evidence_bundle"],
    completed_count: overrides.completed_count ?? 2,
    configured: overrides.configured ?? true,
    failed_count: overrides.failed_count ?? 0,
    process_count: overrides.process_count ?? 0,
    resource_profiles: overrides.resource_profiles ?? [],
    schema_version: overrides.schema_version ?? "studio.jobs.status.v1",
    thread_count: overrides.thread_count ?? 1,
    timed_out_count: overrides.timed_out_count ?? 0,
  };
}

function jobList(overrides: Partial<StudioJobListResponse> = {}): StudioJobListResponse {
  return {
    jobs: overrides.jobs ?? [],
    schema_version: overrides.schema_version ?? "studio.jobs.list.v1",
  };
}

function operatorStatus(overrides: Partial<StudioOperatorStatus> = {}): StudioOperatorStatus {
  return {
    audit: overrides.audit ?? auditStatus(),
    browser_login: overrides.browser_login ?? {
      active_bucket_count: 0,
      cooldown_seconds: 60,
      failure_window_seconds: 300,
      locked_bucket_count: 0,
      max_failures: 5,
      max_retry_after_seconds: 0,
    },
    capabilities: overrides.capabilities ?? {
      degraded_count: 0,
      experimental_count: 0,
      healthy_count: 4,
      stable_count: 4,
      total_count: 4,
      unavailable_count: 0,
    },
    deployment_profile: overrides.deployment_profile ?? "production",
    identity: overrides.identity ?? {
      configured: true,
      header_principal_allowed: false,
      mode: "service_account",
    },
    jobs: overrides.jobs ?? jobStatus(),
    resource_limits: overrides.resource_limits ?? {
      eda_process_cpu_seconds: null,
      eda_process_limits_supported: false,
      eda_process_memory_bytes: null,
      job_default_timeout_seconds: 600,
      job_max_artifact_bytes: 1048576,
    },
    route_policies: overrides.route_policies ?? {
      admin_count: 2,
      authenticated_count: 5,
      enforced: true,
      protected_audit_action_count: 2,
      protected_count: 7,
      protected_routes_audited: true,
      public_count: 3,
      total_count: 10,
    },
    schema_version: overrides.schema_version ?? "studio.operator.status.v1",
  };
}

describe("evidence bundle surface helpers", () => {
  it("maps scoped surfaces to store keys", () => {
    expect(evidenceBundleSurfaceKeys("project")).toEqual({
      bundle: "projectEvidenceBundle",
      error: "projectEvidenceBundleError",
      loading: "projectEvidenceBundleLoading",
    });
    expect(evidenceBundleSurfaceKeys("compile")).toEqual({
      bundle: "compileEvidenceBundle",
      error: "compileEvidenceBundleError",
      loading: "compileEvidenceBundleLoading",
    });
    expect(evidenceBundleSurfaceKeys("synthesis")).toEqual({
      bundle: "synthesisEvidenceBundle",
      error: "synthesisEvidenceBundleError",
      loading: "synthesisEvidenceBundleLoading",
    });
  });

  it("selects the evidence bundle for a scoped surface", () => {
    expect(selectEvidenceBundleForSurface("project", {
      compileEvidenceBundle: null,
      projectEvidenceBundle: bundle,
      synthesisEvidenceBundle: null,
    })).toBe(bundle);
  });

  it("selects the admin evidence bundle download slot", () => {
    expect(evidenceBundleDownloadSelection("admin", {
      compileEvidenceBundle: null,
      evidenceBundle: bundle,
      projectEvidenceBundle: null,
      synthesisEvidenceBundle: null,
    })).toEqual({
      bundle,
      error: "evidenceBundleError",
    });
  });

  it("selects scoped evidence bundle download slots", () => {
    expect(evidenceBundleDownloadSelection("compile", {
      compileEvidenceBundle: bundle,
      evidenceBundle: null,
      projectEvidenceBundle: null,
      synthesisEvidenceBundle: null,
    })).toEqual({
      bundle,
      error: "compileEvidenceBundleError",
    });
  });

  it("chooses the newest synthesis job carrying the requested artefact", () => {
    const jobs = [
      jobRecord("sj_old", "2026-06-21T10:00:00Z", "synthesis", [
        "synthesis/multi-target-result.json",
      ]),
      jobRecord("sj_new", "2026-06-21T11:00:00Z", "synthesis", [
        "synthesis/multi-target-result.json",
      ]),
      jobRecord("sj_compile", "2026-06-21T12:00:00Z", "compiler", [
        "synthesis/multi-target-result.json",
      ]),
    ];

    expect(latestSynthesisJobIdWithArtefact(
      jobs,
      "synthesis/multi-target-result.json",
    )).toBe("sj_new");
  });

  it("returns null when no synthesis job carries the requested artefact", () => {
    expect(latestSynthesisJobIdWithArtefact([
      jobRecord("sj_compile", "2026-06-21T12:00:00Z", "compiler", [
        "compiler/result.json",
      ]),
    ], "synthesis/result.json")).toBeNull();
  });

  it("builds admin evidence bundle state patches", () => {
    const operator = operatorStatus({
      audit: auditStatus({ healthy: false, last_error: "sink warning" }),
      jobs: jobStatus({ completed_count: 3 }),
    });
    const jobs = jobList();

    expect(adminEvidenceBundleLoadingState()).toEqual({
      evidenceBundleError: null,
      evidenceBundleLoading: true,
    });
    expect(adminEvidenceBundleCreatedState(bundle, operator, jobs)).toEqual({
      auditStatus: operator.audit,
      evidenceBundle: bundle,
      evidenceBundleError: null,
      evidenceBundleLoading: false,
      jobRecords: jobs.jobs,
      jobStatus: operator.jobs,
      operatorStatus: operator,
    });
    expect(adminEvidenceBundleFailureState(new Error("bundle failed"))).toEqual({
      evidenceBundleError: "bundle failed",
      evidenceBundleLoading: false,
    });
  });

  it("builds scoped evidence bundle state patches", () => {
    const operator = operatorStatus();
    const jobs = jobList();

    expect(scopedEvidenceBundleLoadingState("project")).toEqual({
      projectEvidenceBundleError: null,
      projectEvidenceBundleLoading: true,
    });
    expect(scopedEvidenceBundleCreatedState("project", bundle, operator, jobs)).toEqual({
      auditStatus: operator.audit,
      jobRecords: jobs.jobs,
      jobStatus: operator.jobs,
      operatorStatus: operator,
      projectEvidenceBundle: bundle,
      projectEvidenceBundleError: null,
      projectEvidenceBundleLoading: false,
    });
    expect(scopedEvidenceBundleFailureState("compile", "bad")).toEqual({
      compileEvidenceBundleError: "Evidence bundle export failed",
      compileEvidenceBundleLoading: false,
    });
  });

  it("builds evidence artifact download error state patches", () => {
    expect(evidenceBundleArtifactUnavailableState("admin")).toEqual({
      evidenceBundleError: "No evidence bundle is available for artifact download.",
    });
    expect(evidenceBundleArtifactDownloadStartState("synthesis")).toEqual({
      synthesisEvidenceBundleError: null,
    });
    expect(
      evidenceBundleArtifactDownloadFailureState("compile", new Error("missing artifact")),
    ).toEqual({ compileEvidenceBundleError: "missing artifact" });
  });

  it("plans an evidence artifact download with scoped state patches", () => {
    const plan = evidenceBundleArtifactDownloadPlan(
      "project",
      "evidence/jobs/sj_bundle/artifacts/manifest.json",
      {
        compileEvidenceBundle: null,
        evidenceBundle: null,
        projectEvidenceBundle: bundle,
        synthesisEvidenceBundle: null,
      },
    );

    expect(plan.available).toBe(true);
    if (!plan.available) {
      throw new Error("expected available evidence bundle artifact plan");
    }

    const downloads: Array<{ payload: Blob; relativePath: string }> = [];
    const payload = new Blob(["manifest"], { type: "application/json" });
    plan.writePayload(payload, (downloadedPayload, relativePath) => {
      downloads.push({ payload: downloadedPayload, relativePath });
    });

    expect(plan.jobId).toBe("sj_bundle");
    expect(plan.relativePath).toBe("evidence/jobs/sj_bundle/artifacts/manifest.json");
    expect(plan.startState).toEqual({ projectEvidenceBundleError: null });
    expect(plan.failureState(new Error("download failed"))).toEqual({
      projectEvidenceBundleError: "download failed",
    });
    expect(downloads).toEqual([{
      payload,
      relativePath: "evidence/jobs/sj_bundle/artifacts/manifest.json",
    }]);
  });

  it("plans an unavailable evidence artifact download without a bundle", () => {
    const plan = evidenceBundleArtifactDownloadPlan(
      "admin",
      "evidence/jobs/sj_bundle/artifacts/manifest.json",
      {
        compileEvidenceBundle: null,
        evidenceBundle: null,
        projectEvidenceBundle: null,
        synthesisEvidenceBundle: null,
      },
    );

    expect(plan).toEqual({
      available: false,
      statePatch: {
        evidenceBundleError: "No evidence bundle is available for artifact download.",
      },
    });
  });
});
