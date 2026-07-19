// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — studioAnalysisJobRunner tests (fake API like W08)
import { afterEach, describe, expect, it, vi } from "vitest";

import type {
  AnalysisJobReceipt,
  AnalysisJobRequestBody,
  FICurveResponse,
  StudioJobRecord,
} from "./api/client";
import type { AnalysisJobApi } from "./analysisJob";
import { createAnalysisJobSession } from "./analysisJob";
import type { StudioSimulationConfigInput } from "./studioSimulationConfig";
import {
  canStartStudioAnalysisJob,
  isAnalysisJobTerminalPhase,
  runStudioAnalysisJob,
} from "./studioAnalysisJobRunner";
import { studioFICurveResultState } from "./studioAnalysisState";

const modelInput: StudioSimulationConfigInput = {
  sourceMode: "model",
  selectedModelName: "lif",
  modelParams: { tau: 10, capacitance: 1 },
  equations: ["dv/dt = -(v - e_l) / tau + i"],
  threshold: "v > -50",
  reset: "v = -65",
  odeParams: { tau: 20, e_l: -65 },
  odeInit: { v: -65 },
  dt: 0.1,
  duration: 100,
  current: 12,
  protocol: "constant",
};

const fiResult: FICurveResponse = {
  analysis_metadata: {
    analysis_type: "fi_curve",
    evidence_classification: "analysis",
    input_sha256: "a".repeat(64),
    output_keys: ["currents", "rates"],
    result_sha256: "b".repeat(64),
    schema_version: "studio.analysis-result.v1",
    source: "model",
    status: "completed",
  },
  currents: [0, 1],
  rates: [0, 5],
};

function jobRecord(
  overrides: Partial<StudioJobRecord> & Pick<StudioJobRecord, "status">,
): StudioJobRecord {
  return {
    artifacts: [],
    created_at_utc: "2026-07-19T00:00:00Z",
    error: null,
    execution_model: "thread",
    finished_at_utc: null,
    job_id: "sj_runner",
    kind: "analysis",
    owner: "studio",
    request_id: null,
    result: null,
    started_at_utc: null,
    ...overrides,
  };
}

function receipt(overrides: Partial<AnalysisJobReceipt> = {}): AnalysisJobReceipt {
  return {
    analysis: "fi_curve",
    execution_mode: "async_job",
    job: jobRecord({ status: "pending" }),
    job_id: "sj_runner",
    schema_version: "studio.analysis.job.v1",
    status_route: "/api/studio/jobs/sj_runner",
    ...overrides,
  };
}

function fakeTimersCreateSession() {
  return (opts: Parameters<typeof createAnalysisJobSession>[0]) =>
    createAnalysisJobSession({
      ...opts,
      setTimeoutFn: setTimeout as typeof setTimeout,
      clearTimeoutFn: clearTimeout as typeof clearTimeout,
    });
}

describe("runStudioAnalysisJob", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("rejects invalid request without calling the API", async () => {
    const submit = vi.fn();
    const fetchJob = vi.fn();
    const applyPatch = vi.fn();
    const outcome = await runStudioAnalysisJob(
      {
        simulation: { ...modelInput, dt: Number.NaN },
        selection: { analysis: "fi_curve" },
      },
      {
        api: { submit, fetchJob },
        applyPatch,
      },
    );
    expect(outcome.ok).toBe(false);
    if (outcome.ok) {
      return;
    }
    expect(outcome.stage).toBe("request");
    expect(outcome.error).toBe("analysis_request_dt_invalid");
    expect(outcome.state).toBeNull();
    expect(submit).not.toHaveBeenCalled();
    expect(fetchJob).not.toHaveBeenCalled();
    expect(applyPatch).not.toHaveBeenCalled();
  });

  it("runs pending→running→completed and sinks the fi_curve patch", async () => {
    vi.useFakeTimers();
    const polls: StudioJobRecord[] = [
      jobRecord({ status: "pending" }),
      jobRecord({ status: "running", started_at_utc: "2026-07-19T00:00:01Z" }),
      jobRecord({
        status: "completed",
        result: { ...fiResult },
        finished_at_utc: "2026-07-19T00:00:02Z",
      }),
    ];
    let idx = 0;
    const submit = vi.fn(async () => receipt());
    const fetchJob = vi.fn(async () => {
      const next = polls[Math.min(idx, polls.length - 1)]!;
      idx += 1;
      return next;
    });
    const api: AnalysisJobApi = { submit, fetchJob };
    const phases: string[] = [];
    const patches: unknown[] = [];

    const runPromise = runStudioAnalysisJob(
      {
        simulation: modelInput,
        selection: { analysis: "fi_curve" },
      },
      {
        api,
        createSession: fakeTimersCreateSession(),
        pollIntervalMs: 20,
        onState: (s) => {
          phases.push(s.phase);
        },
        applyPatch: (p) => {
          patches.push(p);
        },
      },
    );

    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(20);
    await vi.advanceTimersByTimeAsync(20);
    await vi.advanceTimersByTimeAsync(20);
    const outcome = await runPromise;

    expect(outcome.ok).toBe(true);
    if (!outcome.ok) {
      return;
    }
    expect(outcome.kind).toBe("fi_curve");
    expect(outcome.patch).toMatchObject({
      ...studioFICurveResultState(fiResult),
      activeTab: "fi-curve",
      error: null,
    });
    expect(outcome.state.phase).toBe("completed");
    expect(outcome.startPatch.isSimulating).toBe(true);
    expect(outcome.startPatch.activeTab).toBe("fi-curve");
    expect(submit).toHaveBeenCalledTimes(1);
    expect(phases).toContain("pending");
    expect(phases).toContain("running");
    expect(phases).toContain("completed");
    expect(patches[0]).toMatchObject({ isSimulating: true, activeTab: "fi-curve" });
    expect(patches[patches.length - 1]).toMatchObject({
      activeTab: "fi-curve",
      isSimulating: false,
    });
  });

  it("maps malformed completion to session failure without a sink patch", async () => {
    vi.useFakeTimers();
    const submit = vi.fn(async () => receipt());
    const fetchJob = vi.fn(async () =>
      jobRecord({ status: "completed", result: { schema_version: "nope" } }),
    );
    const patches: unknown[] = [];
    const runPromise = runStudioAnalysisJob(
      {
        simulation: modelInput,
        selection: { analysis: "fi_curve" },
      },
      {
        api: { submit, fetchJob },
        createSession: fakeTimersCreateSession(),
        pollIntervalMs: 15,
        applyPatch: (p) => {
          patches.push(p);
        },
      },
    );
    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(15);
    const outcome = await runPromise;
    expect(outcome.ok).toBe(false);
    if (outcome.ok) {
      return;
    }
    expect(outcome.stage).toBe("session");
    expect(outcome.state?.phase).toBe("malformed");
    expect(outcome.failurePatch?.isSimulating).toBe(false);
    expect(patches[0]).toMatchObject({ isSimulating: true });
    expect(patches[patches.length - 1]).toMatchObject({ isSimulating: false });
  });

  it("disposes the binding even when submit fails", async () => {
    vi.useFakeTimers();
    let disposed = false;
    const createSession = vi.fn(
      (opts: Parameters<typeof createAnalysisJobSession>[0]) => {
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
          startJob: async (request: AnalysisJobRequestBody) => {
            await session.startJob(request);
          },
        };
      },
    );
    const runPromise = runStudioAnalysisJob(
      {
        simulation: modelInput,
        selection: { analysis: "fi_curve" },
      },
      {
        createSession,
        api: {
          submit: async () => {
            throw new Error("network_down");
          },
          fetchJob: async () => jobRecord({ status: "pending" }),
        },
        pollIntervalMs: 10,
      },
    );
    await vi.advanceTimersByTimeAsync(0);
    const outcome = await runPromise;
    expect(outcome.ok).toBe(false);
    if (outcome.ok) {
      return;
    }
    expect(outcome.error).toBe("network_down");
    expect(disposed).toBe(true);
  });

  it("rejects completed result that fails the sink shape check", async () => {
    vi.useFakeTimers();
    // Passes W07 job validation if shaped carefully — use metadata kind mismatch
    // via completed result with wrong analysis_type but valid-looking fields.
    // Session may mark completed; sink must fail-closed on kind mismatch.
    const mismatched = {
      analysis_metadata: {
        ...fiResult.analysis_metadata,
        analysis_type: "heatmap",
      },
      currents: [0],
      rates: [1],
    };
    const submit = vi.fn(async () => receipt());
    const fetchJob = vi.fn(async () =>
      jobRecord({
        status: "completed",
        result: mismatched,
        finished_at_utc: "2026-07-19T00:00:02Z",
      }),
    );
    const runPromise = runStudioAnalysisJob(
      {
        simulation: modelInput,
        selection: { analysis: "fi_curve" },
      },
      {
        api: { submit, fetchJob },
        createSession: fakeTimersCreateSession(),
        pollIntervalMs: 10,
      },
    );
    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(10);
    const outcome = await runPromise;
    // Either session malformed (W07 validate) or sink kind mismatch — both ok:false
    expect(outcome.ok).toBe(false);
    if (outcome.ok) {
      return;
    }
    expect(outcome.stage).toBe("session");
    expect(outcome.failurePatch?.isSimulating).toBe(false);
  });
});

describe("runner helpers", () => {
  it("classifies terminal phases and idle can-start", () => {
    expect(isAnalysisJobTerminalPhase("completed")).toBe(true);
    expect(isAnalysisJobTerminalPhase("malformed")).toBe(true);
    expect(isAnalysisJobTerminalPhase("running")).toBe(false);
    expect(canStartStudioAnalysisJob()).toBe(true);
    expect(
      canStartStudioAnalysisJob({
        analysis: "fi_curve",
        error: null,
        jobId: "x",
        phase: "running",
        result: null,
        statusRoute: "/x",
      }),
    ).toBe(false);
  });
});
