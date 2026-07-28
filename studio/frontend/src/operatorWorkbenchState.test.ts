// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio operator workbench state tests
import { describe, expect, it } from "vitest";

import type { SimulateResponse, StudioOperatorStatus } from "./api/client";
import { computeGuidedFlowState, type GuidedFlowInputs } from "./guidedFlowState";
import { buildOperatorWorkbenchState, type OperatorWorkbenchInputs } from "./operatorWorkbenchState";

function guidedInputs(overrides: Partial<GuidedFlowInputs> = {}): GuidedFlowInputs {
  return {
    analysisComplete: false,
    compileComplete: false,
    cosimApplicable: false,
    cosimComplete: false,
    evidenceExported: false,
    modelSelected: false,
    simulationComplete: false,
    synthesisComplete: false,
    trainingComplete: false,
    trainingSkipped: false,
    ...overrides,
  };
}

function operatorStatus(overrides: Partial<StudioOperatorStatus> = {}): StudioOperatorStatus {
  return {
    audit: overrides.audit ?? {
      configured: true,
      healthy: true,
      last_error: null,
      path_configured: true,
      sink_type: "jsonl",
    },
    browser_login: overrides.browser_login ?? {
      active_bucket_count: 0,
      cooldown_seconds: 900,
      failure_window_seconds: 300,
      locked_bucket_count: 0,
      max_failures: 5,
      max_retry_after_seconds: 0,
    },
    capabilities: overrides.capabilities ?? {
      degraded_count: 0,
      experimental_count: 0,
      healthy_count: 8,
      stable_count: 8,
      total_count: 8,
      unavailable_count: 0,
    },
    deployment_profile: overrides.deployment_profile ?? "development",
    identity: overrides.identity ?? {
      configured: true,
      header_principal_allowed: false,
      mode: "service_account",
    },
    jobs: overrides.jobs ?? {
      active_count: 0,
      allowed_kinds: ["evidence_bundle"],
      completed_count: 1,
      configured: true,
      failed_count: 0,
      process_count: 0,
      resource_profiles: [],
      schema_version: "studio.jobs.status.v1",
      thread_count: 0,
      timed_out_count: 0,
    },
    resource_limits: overrides.resource_limits ?? {
      eda_process_cpu_seconds: 120,
      eda_process_limits_supported: true,
      eda_process_memory_bytes: 536870912,
      job_default_timeout_seconds: 600,
      job_max_artifact_bytes: 16777216,
    },
    route_policies: overrides.route_policies ?? {
      admin_count: 1,
      authenticated_count: 2,
      enforced: true,
      protected_audit_action_count: 3,
      protected_count: 3,
      protected_routes_audited: true,
      public_count: 1,
      total_count: 4,
    },
    schema_version: overrides.schema_version ?? "studio.operator.status.v1",
  };
}

function simulationResult(): SimulateResponse {
  return {
    current_trace: [10, 10],
    dt: 0.1,
    model_name: "LIFNeuron",
    n_steps: 2,
    run_metadata: {
      dt: 0.1,
      evidence_classification: "simulation",
      input_sha256: "a".repeat(64),
      n_steps: 2,
      result_sha256: "b".repeat(64),
      sample_count: 2,
      schema_version: "studio.simulation-run.v1",
      source: "model",
      spike_count: 3,
      status: "completed",
      state_variables: ["v"],
    },
    spike_count: 3,
    spikes: [0.1, 0.2, 0.3],
    stats: {
      isi_cv: null,
      isi_histogram: null,
      isi_mean_ms: null,
      rate_hz: 30,
    },
    states: { v: [-65, -50] },
    time: [0, 0.1],
  };
}

function inputs(overrides: Partial<OperatorWorkbenchInputs> = {}): OperatorWorkbenchInputs {
  return {
    compileBundleExported: false,
    compileComplete: false,
    guidedFlow: computeGuidedFlowState(guidedInputs()),
    isSimulating: false,
    modelCount: 118,
    operatorStatus: operatorStatus(),
    progressMessage: "",
    projectBundleExported: false,
    projectName: null,
    savedSessionCount: 0,
    selectedModelName: "",
    serverProjectCount: 0,
    simulationResult: null,
    sourceMode: "model",
    synthesisBundleExported: false,
    synthesisComplete: false,
    ...overrides,
  };
}

describe("buildOperatorWorkbenchState", () => {
  it("surfaces an unsaved initial workspace and blocked run/export actions", () => {
    const state = buildOperatorWorkbenchState(inputs());

    expect(state.headline).toBe("Next: Design");
    expect(state.subhead).toBe("0/7 lifecycle steps complete");
    expect(state.evidenceActionEnabled).toBe(false);
    expect(state.evidenceExportTarget).toBeNull();
    expect(state.cards.map((card) => card.key)).toEqual([
      "workspace",
      "model",
      "simulation",
      "evidence",
      "compile",
      "export",
    ]);
    expect(state.cards.find((card) => card.key === "workspace")).toMatchObject({
      status: "warning",
      value: "Unsaved",
    });
    expect(state.cards.find((card) => card.key === "simulation")).toMatchObject({
      action: "Run simulation",
      status: "blocked",
    });
  });

  it("reports completed simulation, healthy evidence, and export readiness", () => {
    const state = buildOperatorWorkbenchState(inputs({
      guidedFlow: computeGuidedFlowState(guidedInputs({
        analysisComplete: true,
        compileComplete: true,
        modelSelected: true,
        simulationComplete: true,
        trainingSkipped: true,
      })),
      projectName: "bench-project",
      savedSessionCount: 2,
      selectedModelName: "LIFNeuron",
      serverProjectCount: 1,
      simulationResult: simulationResult(),
    }));

    expect(state.headline).toBe("Next: Synthesise");
    expect(state.evidenceActionEnabled).toBe(true);
    expect(state.evidenceExportTarget).toBe("project");
    expect(state.cards.find((card) => card.key === "simulation")).toMatchObject({
      detail: "3 spikes, 30 Hz",
      status: "ready",
      value: "completed",
    });
    expect(state.cards.find((card) => card.key === "evidence")).toMatchObject({
      detail: "8/8 capabilities healthy",
      status: "ready",
    });
    expect(state.cards.find((card) => card.key === "export")).toMatchObject({
      action: "Export project bundle",
      status: "ready",
      value: "project scope",
    });
  });

  it("prefers compile evidence once compile traceability exists", () => {
    const state = buildOperatorWorkbenchState(inputs({
      compileComplete: true,
      projectBundleExported: true,
      projectName: "bench-project",
    }));

    expect(state.evidenceActionEnabled).toBe(true);
    expect(state.evidenceExportTarget).toBe("compile");
    expect(state.cards.find((card) => card.key === "export")).toMatchObject({
      action: "Export compile bundle",
      detail: "Bundle compile traceability, audit excerpt, and RTL provenance",
      status: "ready",
      value: "compile scope",
    });
  });

  it("prefers synthesis evidence and opens it when the bundle already exists", () => {
    const state = buildOperatorWorkbenchState(inputs({
      compileBundleExported: true,
      compileComplete: true,
      projectName: "bench-project",
      synthesisBundleExported: true,
      synthesisComplete: true,
    }));

    expect(state.evidenceActionEnabled).toBe(true);
    expect(state.evidenceExportTarget).toBe("synthesis");
    expect(state.cards.find((card) => card.key === "export")).toMatchObject({
      action: "Open synthesis bundle",
      detail: "synthesis evidence bundle is ready for artifact download",
      status: "ready",
      value: "Bundle ready",
    });
  });

  it("warns when operator evidence health is degraded", () => {
    const degraded = operatorStatus({
      audit: {
        configured: true,
        healthy: false,
        last_error: "append failed",
        path_configured: true,
        sink_type: "jsonl",
      },
      capabilities: {
        degraded_count: 1,
        experimental_count: 0,
        healthy_count: 7,
        stable_count: 7,
        total_count: 8,
        unavailable_count: 1,
      },
    });

    const state = buildOperatorWorkbenchState(inputs({ operatorStatus: degraded }));

    expect(state.cards.find((card) => card.key === "evidence")).toMatchObject({
      action: "Open admin",
      status: "warning",
      value: "development",
    });
  });
});
