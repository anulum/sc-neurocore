// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Model-scan job policy tests
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ModelScanJobReceipt, StudioJobRecord } from "./api/client";
import {
  canSubmitModelScanJob,
  createModelScanJobSession,
  initialModelScanJobState,
  isModelScanJobBusy,
  modelScanJobPhaseLabel,
  reduceModelScanJob,
  validateModelScanJobResult,
} from "./modelScanJob";

function jobRecord(
  overrides: Partial<StudioJobRecord> & Pick<StudioJobRecord, "status">,
): StudioJobRecord {
  return {
    artifacts: [],
    created_at_utc: "2026-07-19T00:00:00Z",
    error: null,
    execution_model: "thread",
    finished_at_utc: null,
    job_id: "sj_scan_1",
    kind: "model_scan",
    owner: "studio",
    request_id: null,
    result: null,
    started_at_utc: null,
    ...overrides,
  };
}

function receipt(
  overrides: Partial<ModelScanJobReceipt> = {},
): ModelScanJobReceipt {
  return {
    execution_mode: "async_job",
    job: jobRecord({ status: "pending" }),
    job_id: "sj_scan_1",
    schema_version: "studio.model-scan.job.v1",
    status_route: "/api/studio/jobs/sj_scan_1",
    ...overrides,
  };
}

const validResult = {
  models: [
    {
      name: "LIFNeuron",
      category: "IF",
      pattern: "tonic",
      description: "tonic",
      rate_hz: 5,
      spike_count: 1,
    },
  ],
  scan_metadata: {
    current: 10,
    duration: 100,
    error_count: 0,
    evidence_classification: "analysis" as const,
    failed_models: [],
    input_sha256: "a".repeat(64),
    model_count: 1,
    pattern_counts: { tonic: 1 },
    result_sha256: "b".repeat(64),
    schema_version: "studio.model-scan.v1" as const,
    status: "completed" as const,
  },
  schema_version: "studio.model-scan.v1" as const,
};

describe("validateModelScanJobResult", () => {
  it("accepts studio.model-scan.v1 with analysis completed metadata", () => {
    const validated = validateModelScanJobResult(validResult);
    expect(validated.ok).toBe(true);
    if (!validated.ok) {
      return;
    }
    expect(validated.value.scan_metadata.evidence_classification).toBe("analysis");
    expect(validated.value.scan_metadata.status).toBe("completed");
  });

  it("rejects missing schema, wrong evidence class, and non-completed status", () => {
    expect(validateModelScanJobResult(null).ok).toBe(false);
    expect(validateModelScanJobResult({ schema_version: "other" }).ok).toBe(false);
    expect(
      validateModelScanJobResult({
        ...validResult,
        scan_metadata: {
          ...validResult.scan_metadata,
          evidence_classification: "simulation",
        },
      }).ok,
    ).toBe(false);
    expect(
      validateModelScanJobResult({
        ...validResult,
        scan_metadata: {
          ...validResult.scan_metadata,
          status: "running",
        },
      }).ok,
    ).toBe(false);
  });
});

describe("reduceModelScanJob", () => {
  it("clears stale success on new submit and blocks duplicates while busy", () => {
    let state = reduceModelScanJob(initialModelScanJobState(), { type: "submit_started" });
    state = reduceModelScanJob(state, {
      type: "submit_succeeded",
      receipt: receipt(),
    });
    state = reduceModelScanJob(state, {
      type: "poll",
      record: jobRecord({
        status: "completed",
        result: validResult,
        finished_at_utc: "2026-07-19T00:01:00Z",
      }),
    });
    expect(state.phase).toBe("completed");
    expect(state.scanMetadata).not.toBeNull();
    expect(Object.keys(state.behaviors)).toEqual(["LIFNeuron"]);

    const restarted = reduceModelScanJob(state, { type: "submit_started" });
    expect(restarted.phase).toBe("submitting");
    expect(restarted.scanMetadata).toBeNull();
    expect(restarted.behaviors).toEqual({});
    expect(isModelScanJobBusy(restarted.phase)).toBe(true);
    expect(canSubmitModelScanJob(restarted)).toBe(false);

    const blocked = reduceModelScanJob(restarted, { type: "submit_started" });
    expect(blocked.phase).toBe("submitting");
  });

  it("fails visibly on terminal non-success and malformed completion without success metadata", () => {
    let state = reduceModelScanJob(initialModelScanJobState(), {
      type: "submit_started",
    });
    state = reduceModelScanJob(state, {
      type: "submit_succeeded",
      receipt: receipt(),
    });
    const failed = reduceModelScanJob(state, {
      type: "poll",
      record: jobRecord({ status: "failed", error: "budget_exceeded" }),
    });
    expect(failed.phase).toBe("failed");
    expect(failed.error).toBe("budget_exceeded");
    expect(failed.scanMetadata).toBeNull();

    const malformed = reduceModelScanJob(state, {
      type: "poll",
      record: jobRecord({
        status: "completed",
        result: { schema_version: "nope" },
        finished_at_utc: "2026-07-19T00:01:00Z",
      }),
    });
    expect(malformed.phase).toBe("malformed");
    expect(malformed.scanMetadata).toBeNull();
    expect(malformed.behaviors).toEqual({});
  });

  it("rejects poll records for a different job id or kind as malformed", () => {
    let state = reduceModelScanJob(initialModelScanJobState(), { type: "submit_started" });
    state = reduceModelScanJob(state, {
      type: "submit_succeeded",
      receipt: receipt(),
    });
    const wrongId = reduceModelScanJob(state, {
      type: "poll",
      record: jobRecord({ status: "running", job_id: "sj_other" }),
    });
    expect(wrongId.phase).toBe("malformed");
    expect(wrongId.error).toBe("model_scan_poll_job_id_mismatch");
    expect(wrongId.scanMetadata).toBeNull();

    const wrongKind = reduceModelScanJob(state, {
      type: "poll",
      record: jobRecord({ status: "running", kind: "analysis" }),
    });
    expect(wrongKind.phase).toBe("malformed");
    expect(wrongKind.error).toBe("model_scan_poll_kind_mismatch");
  });

  it("maps job statuses to real phase labels without inventing progress", () => {
    expect(modelScanJobPhaseLabel("pending")).toBe("pending");
    expect(modelScanJobPhaseLabel("running")).toBe("running");
    expect(modelScanJobPhaseLabel("completed")).toBe("Scanned");
    expect(modelScanJobPhaseLabel("timed_out")).toBe("timed_out");
  });
});

describe("createModelScanJobSession polling", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("submits once, polls pending/running to completed, and ignores duplicate starts", async () => {
    vi.useFakeTimers();
    const polls: StudioJobRecord[] = [
      jobRecord({ status: "pending" }),
      jobRecord({ status: "running", started_at_utc: "2026-07-19T00:00:01Z" }),
      jobRecord({
        status: "completed",
        result: validResult,
        finished_at_utc: "2026-07-19T00:00:02Z",
      }),
    ];
    let pollIndex = 0;
    const submit = vi.fn(async () => receipt());
    const fetchJob = vi.fn(async () => {
      const next = polls[Math.min(pollIndex, polls.length - 1)]!;
      pollIndex += 1;
      return next;
    });
    const states: string[] = [];
    const session = createModelScanJobSession({
      api: { submit, fetchJob },
      pollIntervalMs: 100,
      onChange: (s) => {
        states.push(s.phase);
      },
    });

    const first = session.startScan();
    await vi.advanceTimersByTimeAsync(0);
    await first;
    // Immediate poll consumed pending
    expect(session.getState().phase).toBe("pending");
    await session.startScan();
    expect(submit).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(100);
    expect(session.getState().phase).toBe("running");
    await vi.advanceTimersByTimeAsync(100);
    expect(session.getState().phase).toBe("completed");
    expect(session.getState().scanMetadata?.model_count).toBe(1);
    expect(session.getState().behaviors.LIFNeuron?.pattern).toBe("tonic");
    expect(states).toContain("submitting");
    expect(states).toContain("pending");
    expect(states).toContain("running");
    expect(states).toContain("completed");
    session.dispose();
  });

  it("stops on terminal failure and on dispose before completion", async () => {
    vi.useFakeTimers();
    const fetchJob = vi.fn(async () =>
      jobRecord({ status: "failed", error: "scan_failed" }),
    );
    const session = createModelScanJobSession({
      api: {
        submit: async () => receipt(),
        fetchJob,
      },
      pollIntervalMs: 50,
    });
    const run = session.startScan();
    await vi.advanceTimersByTimeAsync(0);
    await run;
    expect(session.getState().phase).toBe("failed");
    expect(session.getState().error).toBe("scan_failed");
    expect(session.getState().scanMetadata).toBeNull();
    session.dispose();

    let calls = 0;
    const hang = createModelScanJobSession({
      api: {
        submit: async () => receipt(),
        fetchJob: async () => {
          calls += 1;
          return jobRecord({ status: "running" });
        },
      },
      pollIntervalMs: 50,
    });
    const pending = hang.startScan();
    await vi.advanceTimersByTimeAsync(0);
    await pending;
    hang.dispose();
    const afterDispose = calls;
    await vi.advanceTimersByTimeAsync(500);
    expect(calls).toBe(afterDispose);
  });

  it("rejects malformed completion and surfaces poll errors path-free", async () => {
    vi.useFakeTimers();
    const session = createModelScanJobSession({
      api: {
        submit: async () => receipt(),
        fetchJob: async () =>
          jobRecord({
            status: "completed",
            result: { schema_version: "studio.model-scan.v1", models: [] },
          }),
      },
      pollIntervalMs: 50,
    });
    const run = session.startScan();
    await vi.advanceTimersByTimeAsync(0);
    await run;
    expect(session.getState().phase).toBe("malformed");
    expect(session.getState().error).toBe("model_scan_metadata_missing");
    session.dispose();

    const pollFail = createModelScanJobSession({
      api: {
        submit: async () => receipt(),
        fetchJob: async () => {
          throw new Error("/home/anulum/secret leaked");
        },
      },
      pollIntervalMs: 50,
    });
    const run2 = pollFail.startScan();
    await vi.advanceTimersByTimeAsync(0);
    await run2;
    expect(pollFail.getState().phase).toBe("failed");
    expect(pollFail.getState().error).toContain("[path]");
    expect(pollFail.getState().error).not.toContain("/home/anulum");
    pollFail.dispose();
  });
});
