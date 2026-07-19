// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — useAnalysisJob adapter tests (binding + hook contract)
import { afterEach, describe, expect, it, vi } from "vitest";

import type { AnalysisJobReceipt, StudioJobRecord } from "./api/client";
import {
  createAnalysisJobSession,
  initialAnalysisJobState,
  type AnalysisJobApi,
  type AnalysisJobSession,
  type AnalysisJobSessionOptions,
  type AnalysisJobViewState,
} from "./analysisJob";
import {
  attachAnalysisJobReactBinding,
  useAnalysisJob,
} from "./useAnalysisJob";

function jobRecord(
  overrides: Partial<StudioJobRecord> & Pick<StudioJobRecord, "status">,
): StudioJobRecord {
  return {
    artifacts: [],
    created_at_utc: "2026-07-19T00:00:00Z",
    error: null,
    execution_model: "thread",
    finished_at_utc: null,
    job_id: "sj_a1",
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
    job_id: "sj_a1",
    schema_version: "studio.analysis.job.v1",
    status_route: "/api/studio/jobs/sj_a1",
    ...overrides,
  };
}

const fiResult = {
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

describe("attachAnalysisJobReactBinding / useAnalysisJob contract", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("starts idle and exposes canSubmit via binding state", () => {
    const states: AnalysisJobViewState[] = [];
    const binding = attachAnalysisJobReactBinding({
      onState: (s) => {
        states.push(s);
      },
      api: {
        submit: async () => receipt(),
        fetchJob: async () => jobRecord({ status: "pending" }),
      },
    });
    expect(binding.getState().phase).toBe("idle");
    expect(binding.getState()).toEqual(initialAnalysisJobState());
    expect(states[0]?.phase).toBe("idle");
    binding.dispose();
  });

  it("submits through pending/running to completed with injected API", async () => {
    vi.useFakeTimers();
    const polls: StudioJobRecord[] = [
      jobRecord({ status: "pending" }),
      jobRecord({ status: "running", started_at_utc: "2026-07-19T00:00:01Z" }),
      jobRecord({
        status: "completed",
        result: fiResult,
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
    const binding = attachAnalysisJobReactBinding({
      api,
      pollIntervalMs: 50,
      onState: (s) => {
        phases.push(s.phase);
      },
    });
    const run = binding.startJob({ analysis: "fi_curve", payload: {} });
    await vi.advanceTimersByTimeAsync(0);
    await Promise.resolve(run);
    expect(binding.getState().phase).toBe("pending");
    await vi.advanceTimersByTimeAsync(50);
    expect(binding.getState().phase).toBe("running");
    await vi.advanceTimersByTimeAsync(50);
    expect(binding.getState().phase).toBe("completed");
    expect(binding.getState().result).not.toBeNull();
    expect(submit).toHaveBeenCalledTimes(1);
    expect(phases).toContain("pending");
    expect(phases).toContain("running");
    expect(phases).toContain("completed");
    binding.dispose();
  });

  it("propagates malformed completion and blocks duplicate submit while busy", async () => {
    vi.useFakeTimers();
    const submit = vi.fn(async () => receipt());
    const fetchJob = vi.fn(async () =>
      jobRecord({ status: "completed", result: { schema_version: "nope" } }),
    );
    const binding = attachAnalysisJobReactBinding({
      api: { submit, fetchJob },
      pollIntervalMs: 20,
    });
    const run = binding.startJob({ analysis: "fi_curve", payload: {} });
    await vi.advanceTimersByTimeAsync(0);
    await Promise.resolve(run);
    expect(binding.getState().phase).toBe("malformed");
    expect(binding.getState().result).toBeNull();
    expect(binding.getState().error).toBeTruthy();
    binding.dispose();

    const busySubmit = vi.fn(async () => {
      await new Promise((resolve) => {
        setTimeout(resolve, 500);
      });
      return receipt();
    });
    const busy = attachAnalysisJobReactBinding({
      api: {
        submit: busySubmit,
        fetchJob: async () => jobRecord({ status: "pending" }),
      },
      pollIntervalMs: 50,
    });
    void busy.startJob({ analysis: "fi_curve", payload: {} });
    void busy.startJob({ analysis: "fi_curve", payload: {} });
    await vi.advanceTimersByTimeAsync(0);
    expect(busySubmit).toHaveBeenCalledTimes(1);
    busy.dispose();
  });

  it("disposes on unmount and ignores stale onChange after dispose", async () => {
    let onChange: ((s: AnalysisJobViewState) => void) | undefined;
    let disposed = false;
    let externalUpdates = 0;
    const fakeSession: AnalysisJobSession = {
      dispose: () => {
        disposed = true;
      },
      getState: () => initialAnalysisJobState(),
      startJob: async () => undefined,
    };
    const createSession = vi.fn((opts: AnalysisJobSessionOptions) => {
      onChange = opts.onChange;
      return fakeSession;
    });
    const binding = attachAnalysisJobReactBinding({
      createSession,
      onState: () => {
        externalUpdates += 1;
      },
      api: {
        submit: async () => receipt(),
        fetchJob: async () => jobRecord({ status: "pending" }),
      },
    });
    expect(createSession).toHaveBeenCalled();
    const updatesBeforeDispose = externalUpdates;
    binding.dispose();
    expect(disposed).toBe(true);
    onChange?.({
      ...initialAnalysisJobState(),
      phase: "running",
      jobId: "sj_stale",
    });
    expect(externalUpdates).toBe(updatesBeforeDispose);
  });

  it("defaults createSession to createAnalysisJobSession and exports useAnalysisJob", () => {
    const createSession = vi.fn((opts: AnalysisJobSessionOptions) =>
      createAnalysisJobSession(opts),
    );
    const binding = attachAnalysisJobReactBinding({
      createSession,
      api: {
        submit: async () => receipt(),
        fetchJob: async () => jobRecord({ status: "pending" }),
      },
    });
    expect(createSession).toHaveBeenCalledTimes(1);
    expect(typeof useAnalysisJob).toBe("function");
    binding.dispose();
  });
});
