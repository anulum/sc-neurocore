// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Analysis job session/policy tests
import { afterEach, describe, expect, it, vi } from "vitest";

import type { AnalysisJobReceipt, StudioJobRecord } from "./api/client";
import {
  canSubmitAnalysisJob,
  createAnalysisJobSession,
  initialAnalysisJobState,
  isAnalysisJobBusy,
  reduceAnalysisJob,
} from "./analysisJob";

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

describe("reduceAnalysisJob", () => {
  it("blocks duplicate submit while busy and clears prior result on restart", () => {
    let state = reduceAnalysisJob(initialAnalysisJobState(), {
      type: "submit_started",
      analysis: "fi_curve",
    });
    state = reduceAnalysisJob(state, {
      type: "submit_succeeded",
      receipt: receipt(),
    });
    state = reduceAnalysisJob(state, {
      type: "poll",
      record: jobRecord({
        status: "completed",
        result: fiResult,
        finished_at_utc: "2026-07-19T00:01:00Z",
      }),
    });
    expect(state.phase).toBe("completed");
    expect(state.result).not.toBeNull();

    const restart = reduceAnalysisJob(state, {
      type: "submit_started",
      analysis: "heatmap",
    });
    expect(restart.phase).toBe("submitting");
    expect(restart.result).toBeNull();
    expect(isAnalysisJobBusy(restart.phase)).toBe(true);
    expect(canSubmitAnalysisJob(restart)).toBe(false);
    expect(
      reduceAnalysisJob(restart, {
        type: "submit_started",
        analysis: "heatmap",
      }).phase,
    ).toBe("submitting");
  });

  it("fails on terminal failure and path-redacts poll errors", () => {
    let state = reduceAnalysisJob(initialAnalysisJobState(), {
      type: "submit_started",
      analysis: "fi_curve",
    });
    state = reduceAnalysisJob(state, {
      type: "submit_succeeded",
      receipt: receipt(),
    });
    const failed = reduceAnalysisJob(state, {
      type: "poll",
      record: jobRecord({ status: "failed", error: "budget_exceeded" }),
    });
    expect(failed.phase).toBe("failed");
    expect(failed.error).toBe("budget_exceeded");
    expect(failed.result).toBeNull();

    const redacted = reduceAnalysisJob(state, {
      type: "poll_failed",
      message: "boom /home/anulum/secret",
    });
    expect(redacted.phase).toBe("failed");
    expect(redacted.error).toContain("[path]");
    expect(redacted.error).not.toContain("/home/anulum");
  });

  it("rejects poll id/kind mismatch as malformed without success result", () => {
    let state = reduceAnalysisJob(initialAnalysisJobState(), {
      type: "submit_started",
      analysis: "fi_curve",
    });
    state = reduceAnalysisJob(state, {
      type: "submit_succeeded",
      receipt: receipt(),
    });
    const wrong = reduceAnalysisJob(state, {
      type: "poll",
      record: jobRecord({ status: "running", job_id: "sj_other" }),
    });
    expect(wrong.phase).toBe("malformed");
    expect(wrong.error).toBe("analysis_poll_job_id_mismatch");
    expect(wrong.result).toBeNull();
  });
});

describe("createAnalysisJobSession", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("submits once, polls pending/running to completed, ignores duplicate starts", async () => {
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
    const session = createAnalysisJobSession({
      api: { submit, fetchJob },
      pollIntervalMs: 100,
    });
    const first = session.startJob({ analysis: "fi_curve", payload: {} });
    await vi.advanceTimersByTimeAsync(0);
    await first;
    expect(session.getState().phase).toBe("pending");
    await session.startJob({ analysis: "fi_curve", payload: {} });
    expect(submit).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(100);
    expect(session.getState().phase).toBe("running");
    await vi.advanceTimersByTimeAsync(100);
    expect(session.getState().phase).toBe("completed");
    expect(session.getState().result).not.toBeNull();
    session.dispose();
  });

  it("stops on terminal failure and on dispose", async () => {
    vi.useFakeTimers();
    const session = createAnalysisJobSession({
      api: {
        submit: async () => receipt(),
        fetchJob: async () => jobRecord({ status: "failed", error: "scan_failed" }),
      },
      pollIntervalMs: 50,
    });
    const run = session.startJob({ analysis: "fi_curve", payload: {} });
    await vi.advanceTimersByTimeAsync(0);
    await run;
    expect(session.getState().phase).toBe("failed");
    expect(session.getState().result).toBeNull();
    session.dispose();

    let calls = 0;
    const hang = createAnalysisJobSession({
      api: {
        submit: async () => receipt(),
        fetchJob: async () => {
          calls += 1;
          return jobRecord({ status: "running" });
        },
      },
      pollIntervalMs: 50,
    });
    const pending = hang.startJob({ analysis: "fi_curve", payload: {} });
    await vi.advanceTimersByTimeAsync(0);
    await pending;
    hang.dispose();
    const after = calls;
    await vi.advanceTimersByTimeAsync(500);
    expect(calls).toBe(after);
  });
});
