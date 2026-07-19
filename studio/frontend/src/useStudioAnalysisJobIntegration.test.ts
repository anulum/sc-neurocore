// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — useStudioAnalysisJobIntegration tests
import { afterEach, describe, expect, it, vi } from "vitest";
import type {
  AnalysisJobReceipt,
  FICurveResponse,
  StudioJobRecord,
} from "./api/client";
import {
  createAnalysisJobSession,
  initialAnalysisJobState,
  type AnalysisJobApi,
  type AnalysisJobViewState,
} from "./analysisJob";
import type { StudioSimulationConfigInput } from "./studioSimulationConfig";
import { attachAnalysisJobReactBinding } from "./useAnalysisJob";
import {
  applyCompletedAnalysisJobResult,
  resolveStudioAnalysisJobIntegration,
  studioAnalysisJobIntegrationCanSubmit,
  useStudioAnalysisJobIntegration,
  type StudioAnalysisJobIntegrationInput,
} from "./useStudioAnalysisJobIntegration";

const modelInput: StudioSimulationConfigInput = {
  sourceMode: "model",
  selectedModelName: "lif",
  modelParams: { tau: 10, capacitance: 1 },
  equations: ["dv/dt = 0"],
  threshold: "v > -50",
  reset: "v = -65",
  odeParams: { tau: 20, e_l: -65 },
  odeInit: { v: -65 },
  dt: 0.1,
  duration: 100,
  current: 12,
  protocol: "constant",
};
const baseInput: StudioAnalysisJobIntegrationInput = {
  simulation: modelInput,
  analysis: "fi_curve",
  sweepParam: "tau",
  sweepParamY: "capacitance",
};
const fiResult: FICurveResponse = {
  analysis_metadata: {
    analysis_type: "fi_curve", evidence_classification: "analysis",
    input_sha256: "a".repeat(64), output_keys: ["currents", "rates"],
    result_sha256: "b".repeat(64), schema_version: "studio.analysis-result.v1",
    source: "model", status: "completed",
  },
  currents: [0, 1], rates: [0, 5],
};

function jobRecord(
  o: Partial<StudioJobRecord> & Pick<StudioJobRecord, "status">,
): StudioJobRecord {
  return {
    artifacts: [], created_at_utc: "2026-07-20T00:00:00Z", error: null,
    execution_model: "thread", finished_at_utc: null, job_id: "sj_int",
    kind: "analysis", owner: "studio", request_id: null, result: null,
    started_at_utc: null, ...o,
  };
}

function receipt(): AnalysisJobReceipt {
  return {
    analysis: "fi_curve", execution_mode: "async_job",
    job: jobRecord({ status: "pending" }), job_id: "sj_int",
    schema_version: "studio.analysis.job.v1",
    status_route: "/api/studio/jobs/sj_int",
  };
}

describe("resolveStudioAnalysisJobIntegration", () => {
  it("builds workbench props; capability and disabled gates", () => {
    const a = resolveStudioAnalysisJobIntegration(baseInput);
    expect(a.selection).toEqual({ analysis: "fi_curve" });
    expect(a.selectedAnalysisLabel).toBe("f-I curve");
    expect(a.request.ok).toBe(true);
    expect(a.disabled).toBe(false);
    expect(a.workbenchProps?.simulationInput).toBe(modelInput);
    expect(
      resolveStudioAnalysisJobIntegration(baseInput, {
        capabilityEnabled: false,
      }).disabled,
    ).toBe(true);
    expect(
      resolveStudioAnalysisJobIntegration(baseInput, { disabled: true }).disabled,
    ).toBe(true);
    expect(
      studioAnalysisJobIntegrationCanSubmit({
        sessionCanSubmit: true,
        disabled: true,
        requestOk: true,
      }),
    ).toBe(false);
    expect(
      studioAnalysisJobIntegrationCanSubmit({
        sessionCanSubmit: true,
        disabled: false,
        requestOk: true,
      }),
    ).toBe(true);
  });

  it("fail-closes invalid bifurcation sweep", () => {
    const r = resolveStudioAnalysisJobIntegration({
      ...baseInput,
      analysis: "bifurcation",
      sweepParam: "   ",
    });
    expect(r.selection).toBeNull();
    expect(r.selectionError).toBe("analysis_selection_sweep_param_blank");
    expect(r.workbenchProps).toBeNull();
    expect(r.request.ok).toBe(false);
  });
});

describe("applyCompletedAnalysisJobResult", () => {
  it("sinks completed, skips idle, fail-closes mismatch", () => {
    const patches: unknown[] = [];
    const completed: AnalysisJobViewState = {
      ...initialAnalysisJobState(),
      analysis: "fi_curve",
      phase: "completed",
      jobId: "sj_int",
      result: fiResult,
    };
    expect(
      applyCompletedAnalysisJobResult({
        kind: "fi_curve",
        state: completed,
        applyPatch: (p) => {
          patches.push(p);
        },
      }),
    ).toEqual({ applied: true, error: null });
    expect(patches[0]).toMatchObject({ activeTab: "fi-curve", isSimulating: false });
    const idle = vi.fn();
    expect(
      applyCompletedAnalysisJobResult({
        kind: "fi_curve",
        state: initialAnalysisJobState(),
        applyPatch: idle,
      }).applied,
    ).toBe(false);
    expect(idle).not.toHaveBeenCalled();
    const bad: FICurveResponse = {
      ...fiResult,
      analysis_metadata: {
        ...fiResult.analysis_metadata,
        analysis_type: "heatmap",
      },
    };
    const fail: unknown[] = [];
    const out = applyCompletedAnalysisJobResult({
      kind: "fi_curve",
      state: { ...initialAnalysisJobState(), phase: "completed", result: bad },
      applyPatch: (p) => {
        fail.push(p);
      },
    });
    expect(out.applied).toBe(false);
    expect(out.error).toContain("kind_mismatch");
    expect(fail[0]).toMatchObject({ isSimulating: false });
  });
});

describe("session dispose + completion sink", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("disposes createSession and sinks completed poll", async () => {
    vi.useFakeTimers();
    let disposed = false;
    let idx = 0;
    const polls: StudioJobRecord[] = [
      jobRecord({ status: "pending" }),
      jobRecord({
        status: "completed",
        result: { ...fiResult },
        finished_at_utc: "2026-07-20T00:00:02Z",
      }),
    ];
    const api: AnalysisJobApi = {
      submit: async () => receipt(),
      fetchJob: async () => {
        const next = polls[Math.min(idx, polls.length - 1)]!;
        idx += 1;
        return next;
      },
    };
    const patches: unknown[] = [];
    const binding = attachAnalysisJobReactBinding({
      api,
      pollIntervalMs: 10,
      createSession: (opts) => {
        const session = createAnalysisJobSession({
          ...opts,
          setTimeoutFn: setTimeout as typeof setTimeout,
          clearTimeoutFn: clearTimeout as typeof clearTimeout,
        });
        return {
          dispose: () => {
            disposed = true;
            session.dispose();
          },
          getState: () => session.getState(),
          startJob: async (request) => {
            await session.startJob(request);
          },
        };
      },
      onState: (s) => {
        if (s.phase === "completed") {
          applyCompletedAnalysisJobResult({
            kind: "fi_curve",
            state: s,
            applyPatch: (p) => {
              patches.push(p);
            },
          });
        }
      },
    });
    const resolved = resolveStudioAnalysisJobIntegration(baseInput);
    expect(resolved.request.ok).toBe(true);
    if (!resolved.request.ok) return;
    binding.startJob(resolved.request.value);
    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(20);
    expect(patches[0]).toMatchObject({ activeTab: "fi-curve" });
    binding.dispose();
    expect(disposed).toBe(true);
    expect(typeof useStudioAnalysisJobIntegration).toBe("function");
  });
});
