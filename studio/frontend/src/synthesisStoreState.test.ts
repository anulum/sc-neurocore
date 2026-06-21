// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio synthesis store state helper tests
import { describe, expect, it } from "vitest";

import type {
  MultiTargetResult,
  StudioAuditStatus,
  StudioJobListResponse,
  StudioJobRecord,
  StudioJobStatus,
  StudioOperatorStatus,
  SynthCapacity,
  SynthEstimate,
  SynthResources,
  SynthResult,
} from "./api/client";
import {
  multiTargetSynthesisRunCompletedState,
  multiTargetSynthesisRunStartState,
  synthesisErrorMessageState,
  synthesisErrorState,
  synthesisEstimateLoadedState,
  synthesisFailureState,
  synthesisRunCompletedState,
  synthesisRunStartState,
  synthesisTargetState,
  synthesisToolStatusLoadedState,
} from "./synthesisStoreState";

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
    active_count: overrides.active_count ?? 0,
    allowed_kinds: overrides.allowed_kinds ?? ["synthesis"],
    completed_count: overrides.completed_count ?? 2,
    configured: overrides.configured ?? true,
    failed_count: overrides.failed_count ?? 0,
    process_count: overrides.process_count ?? 0,
    resource_profiles: overrides.resource_profiles ?? [],
    schema_version: overrides.schema_version ?? "studio.jobs.status.v1",
    thread_count: overrides.thread_count ?? 0,
    timed_out_count: overrides.timed_out_count ?? 0,
  };
}

function jobRecord(
  jobId: string,
  createdAt: string,
  artifactPath: string,
): StudioJobRecord {
  return {
    artifacts: [{
      relative_path: artifactPath,
      sha256: "f".repeat(64),
      size_bytes: 64,
    }],
    created_at_utc: createdAt,
    error: null,
    execution_model: "process",
    finished_at_utc: createdAt,
    job_id: jobId,
    kind: "synthesis",
    owner: "studio-synthesis",
    request_id: null,
    result: null,
    started_at_utc: createdAt,
    status: "completed",
  };
}

function jobList(overrides: Partial<StudioJobListResponse> = {}): StudioJobListResponse {
  return {
    jobs: overrides.jobs ?? [],
    schema_version: overrides.schema_version ?? "studio.jobs.list.v1",
  };
}

function capacity(overrides: Partial<SynthCapacity> = {}): SynthCapacity {
  return {
    brams: overrides.brams ?? 8,
    dsps: overrides.dsps ?? 16,
    ffs: overrides.ffs ?? 2000,
    luts: overrides.luts ?? 1000,
  };
}

function resources(overrides: Partial<SynthResources> = {}): SynthResources {
  return {
    brams: overrides.brams ?? 1,
    cells: overrides.cells ?? 12,
    dsps: overrides.dsps ?? 0,
    ffs: overrides.ffs ?? 20,
    luts: overrides.luts ?? 10,
    wires: overrides.wires ?? 30,
  };
}

function synthResult(overrides: Partial<SynthResult> = {}): SynthResult {
  return {
    capacity: overrides.capacity ?? capacity(),
    error: overrides.error,
    log_excerpt: overrides.log_excerpt ?? "synthesis complete",
    resources: overrides.resources ?? resources(),
    success: overrides.success ?? true,
    target: overrides.target ?? "ice40",
    target_provenance: overrides.target_provenance ?? {
      capacity: capacity(),
      device: "hx8k",
      evidence_classification: "synthesis",
      pnr_ready: true,
      pnr_tool: "nextpnr",
      schema_version: "studio.synthesis-target-provenance.v1",
      synthesis_command: "synth_ice40",
      synthesis_ready: true,
      target: "ice40",
      tools: [],
    },
    utilisation: overrides.utilisation ?? { luts: 0.01 },
  };
}

function multiTargetResult(
  overrides: Partial<MultiTargetResult> = {},
): MultiTargetResult {
  return {
    supported: overrides.supported ?? ["ice40"],
    target_provenance_matrix: overrides.target_provenance_matrix ?? {
      matrix_sha256: "a".repeat(64),
      schema_version: "studio.synthesis-target-provenance-matrix.v1",
      targets: { ice40: synthResult().target_provenance },
    },
    targets: overrides.targets ?? { ice40: synthResult() },
  };
}

function synthEstimate(overrides: Partial<SynthEstimate> = {}): SynthEstimate {
  return {
    capacity: overrides.capacity ?? capacity(),
    estimated: overrides.estimated ?? true,
    resources: overrides.resources ?? { brams: 1, dsps: 0, ffs: 20, luts: 10 },
    target: overrides.target ?? "ice40",
    utilisation: overrides.utilisation ?? { luts: 0.01 },
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

describe("synthesis store state helpers", () => {
  it("builds single-target and multi-target start patches", () => {
    expect(synthesisRunStartState()).toEqual({
      activeTab: "synth",
      error: null,
      isSimulating: true,
      latestSynthesisJobId: null,
      multiTargetResult: null,
      synthesisEvidenceBundle: null,
      synthesisEvidenceBundleError: null,
    });
    expect(multiTargetSynthesisRunStartState()).toEqual({
      activeTab: "synth",
      error: null,
      isSimulating: true,
      latestMultiTargetSynthesisJobId: null,
      synthResult: null,
      synthesisEvidenceBundle: null,
      synthesisEvidenceBundleError: null,
    });
  });

  it("builds the single-target completion patch with the newest artifact job", () => {
    const result = synthResult();
    const operator = operatorStatus({ jobs: jobStatus({ completed_count: 3 }) });
    const jobs = jobList({
      jobs: [
        jobRecord("sj_old", "2026-06-21T08:00:00Z", "synthesis/result.json"),
        jobRecord("sj_new", "2026-06-21T09:00:00Z", "synthesis/result.json"),
      ],
    });

    expect(synthesisRunCompletedState(result, operator, jobs)).toEqual({
      auditStatus: operator.audit,
      isSimulating: false,
      jobRecords: jobs.jobs,
      jobStatus: operator.jobs,
      latestSynthesisJobId: "sj_new",
      operatorStatus: operator,
      synthResult: result,
    });
  });

  it("builds the multi-target completion patch with the newest artifact job", () => {
    const result = multiTargetResult();
    const operator = operatorStatus();
    const jobs = jobList({
      jobs: [
        jobRecord("sj_single", "2026-06-21T09:00:00Z", "synthesis/result.json"),
        jobRecord("sj_multi", "2026-06-21T09:01:00Z", "synthesis/multi-target-result.json"),
      ],
    });

    expect(multiTargetSynthesisRunCompletedState(result, operator, jobs)).toEqual({
      auditStatus: operator.audit,
      isSimulating: false,
      jobRecords: jobs.jobs,
      jobStatus: operator.jobs,
      latestMultiTargetSynthesisJobId: "sj_multi",
      multiTargetResult: result,
      operatorStatus: operator,
    });
  });

  it("builds failure, target, estimate, and tool-status patches", () => {
    const estimate = synthEstimate();

    expect(synthesisFailureState(new Error("synth offline"))).toEqual({
      error: "synth offline",
      isSimulating: false,
    });
    expect(synthesisFailureState("bad")).toEqual({
      error: "Synthesis failed",
      isSimulating: false,
    });
    expect(synthesisErrorState(new Error("estimate offline"), "fallback")).toEqual({
      error: "estimate offline",
    });
    expect(synthesisErrorState("bad", "fallback")).toEqual({ error: "fallback" });
    expect(synthesisErrorMessageState("Generate Verilog first")).toEqual({
      error: "Generate Verilog first",
    });
    expect(synthesisEstimateLoadedState(estimate)).toEqual({ synthEstimate: estimate });
    expect(synthesisTargetState("ecp5")).toEqual({ synthTarget: "ecp5" });
    expect(synthesisToolStatusLoadedState({ yosys: { available: true, version: "0.50" } }))
      .toEqual({ toolsAvailable: { yosys: { available: true, version: "0.50" } } });
  });
});
