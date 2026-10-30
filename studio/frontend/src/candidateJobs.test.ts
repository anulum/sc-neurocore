// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Candidate job protocol negative controls

import { describe, expect, it } from "vitest";
import { bindCandidateJobObservation, candidateJobActive, parseCandidateJob, parseCandidateJobHistory, parseCandidateJobReceipt } from "./candidateJobs";

const HASH = "a".repeat(64);
const REQUEST = "b".repeat(64);
const ID = "sj_0123456789abcdef";

/** Build a complete protocol specimen for parser negative controls. */
function specimen(operation = "simulate"): Record<string, unknown> {
  return {
    artifacts: [{ relative_path: "result.json", sha256: HASH, size_bytes: 1 }],
    created_at_utc: "2026-10-30T00:00:00Z", error: null, execution_model: "process",
    finished_at_utc: "2026-10-30T00:01:00Z", job_id: ID, kind: "analysis", owner: "alice",
    request_id: null, started_at_utc: "2026-10-30T00:00:01Z", status: "completed",
    experiment_sha256: REQUEST,
    admission: { candidate_operation: operation, candidate_name: "Candidate", candidate_sha256: HASH,
      request_sha256: REQUEST, projected_steps: 2 },
    result: operation === "simulate" ? {
      candidate: "Candidate", candidate_sha256: HASH, units: { current: "pA", time: "ms" },
      steps: 2, current: 0, profile: { method: "euler", dt: 0.1, time_unit: "ms" },
      spike_count: 1, spike_steps: [1], final_state: { v: -65 }, diverged_at_step: null,
      divergence: null, sample_every: 1, trace: { v: [-65, -64] },
    } : {
      schema_version: "sc-neurocore.studio.candidate-review.v1", candidate_sha256: HASH,
      packet_sha256: REQUEST, reference_tests_passed: false, not_established: ["catalogue promotion"],
      reference_tests: [{ name: "rest", passed: false, diverged_at_step: 0,
        checks: [{ quantity: "final_state.v", observed: null, held: false }] }],
    },
  };
}

/** Substitute one JSON field without asserting a malformed value as a valid record. */
function substitute(value: Record<string, unknown>, path: string, replacement: unknown): void {
  const parts = path.split(".");
  let parent: unknown = value;
  for (const key of parts.slice(0, -1)) {
    if (typeof parent !== "object" || parent === null) throw new Error("Invalid test path.");
    parent = Reflect.get(parent, key);
  }
  if (typeof parent !== "object" || parent === null) throw new Error("Invalid test parent.");
  Reflect.set(parent, parts.at(-1) ?? "", replacement);
}

describe("candidate job protocol", () => {
  it("admits both scientific operations and explicit interrupted recovery states", () => {
    expect(parseCandidateJob(specimen(), ID).result).toHaveProperty("spike_count", 1);
    expect(parseCandidateJob(specimen("review")).result).toHaveProperty("reference_tests_passed", false);
    for (const status of ["pending", "running", "cancelling", "failed", "cancelled", "timed_out", "interrupted", "unknown"]) {
      const job = parseCandidateJob({ ...specimen(), status, result: null });
      expect(candidateJobActive(job)).toBe(["pending", "running", "cancelling"].includes(status));
    }
    const diverged = specimen();
    substitute(diverged, "result.final_state", null);
    substitute(diverged, "result.diverged_at_step", 1);
    substitute(diverged, "result.divergence", "non-finite state");
    expect(parseCandidateJob(diverged).result).toHaveProperty("final_state", null);
    const passed = specimen("review");
    substitute(passed, "result.reference_tests.0.diverged_at_step", null);
    substitute(passed, "result.reference_tests.0.checks.0.observed", -65);
    expect(parseCandidateJob(passed).result).not.toBeNull();
  });

  it.each([
    ["job_id", "foreign"], ["execution_model", "thread"], ["kind", "compiler"], ["status", "guessed"],
    ["admission", null], ["admission.candidate_operation", "other"], ["admission.candidate_name", 3],
    ["admission.candidate_name", ""], ["admission.candidate_sha256", null], ["admission.request_sha256", "bad"],
    ["admission.projected_steps", -1], ["experiment_sha256", HASH], ["result", null], ["result", []],
    ["result.candidate", null], ["result.candidate_sha256", REQUEST], ["result.units", null],
    ["result.units.current", false], ["result.units.time", 2], ["result.steps", 0], ["result.steps", 0.5],
    ["result.current", Infinity], ["result.profile", null], ["result.profile.method", 3],
    ["result.profile.dt", 0], ["result.profile.time_unit", false], ["result.spike_count", -1],
    ["result.spike_steps", null], ["result.spike_steps", [-1]], ["result.spike_count", 2],
    ["result.final_state", []], ["result.final_state.v", NaN], ["result.diverged_at_step", -1],
    ["result.divergence", false], ["result.sample_every", 0], ["result.trace", null],
    ["result.trace.v", 1], ["result.trace.v", [Infinity]],
  ])("refuses substituted simulation field %s", (path, replacement) => {
    const value = specimen();
    substitute(value, path, replacement);
    expect(() => parseCandidateJob(value)).toThrow();
  });

  it.each([
    ["result", null], ["result", false], ["result.schema_version", "other"], ["result.packet_sha256", "bad"],
    ["result.reference_tests_passed", 1], ["result.not_established", null], ["result.not_established", [3]],
    ["result.reference_tests", null], ["result.reference_tests", [null]], ["result.reference_tests.0.name", 1],
    ["result.reference_tests.0.passed", null], ["result.reference_tests.0.diverged_at_step", "unknown"],
    ["result.reference_tests.0.checks", null], ["result.reference_tests.0.checks", [null]],
    ["result.reference_tests.0.checks.0.quantity", 0], ["result.reference_tests.0.checks.0.observed", NaN],
    ["result.reference_tests.0.checks.0.held", 1],
  ])("refuses substituted review field %s", (path, replacement) => {
    const value = specimen("review");
    substitute(value, path, replacement);
    expect(() => parseCandidateJob(value)).toThrow();
  });

  it("binds receipt and history without requiring summary result arrays", () => {
    const value = { schema_version: "studio.candidate.job.v1", operation: "simulate", job_id: ID,
      candidate_sha256: HASH, request_sha256: REQUEST, projected_steps: 2,
      status_route: `/api/candidates/jobs/${ID}`, job: specimen() };
    expect(parseCandidateJobReceipt(value, "simulate").job.job_id).toBe(ID);
    for (const [key, replacement] of Object.entries({ operation: "review", job_id: "sj_0000000000000000",
      candidate_sha256: REQUEST, request_sha256: HASH, projected_steps: 3, status_route: "/other" })) {
      expect(() => parseCandidateJobReceipt({ ...value, [key]: replacement }, "simulate")).toThrow();
    }
    expect(() => parseCandidateJobReceipt(null, "simulate")).toThrow();
    expect(() => parseCandidateJobReceipt({ ...value, schema_version: "other" }, "simulate")).toThrow();
    expect(() => parseCandidateJobReceipt({ ...value, operation: "review" }, "review")).toThrow();
    expect(() => parseCandidateJob(specimen(), "sj_0000000000000000")).toThrow();
    const history = { schema_version: "studio.candidate.jobs.v1", jobs: [{ ...specimen(), result: null }] };
    expect(parseCandidateJobHistory(history)[0]?.result).toBeNull();
    for (const malformed of [null, {}, { ...history, jobs: null }, { ...history, jobs: Array(101).fill(specimen()) }]) {
      expect(() => parseCandidateJobHistory(malformed)).toThrow();
    }
  });

  it("refuses contradictory final-state and divergence claims", () => {
    for (const values of [
      { final_state: null, diverged_at_step: null, divergence: null },
      { final_state: { v: -65 }, diverged_at_step: 1, divergence: "overflow" },
      { final_state: null, diverged_at_step: 1, divergence: null },
      { final_state: null, diverged_at_step: -1, divergence: "overflow" },
      { final_state: { v: -65 }, diverged_at_step: null, divergence: "overflow" },
    ]) {
      const document = specimen();
      for (const [name, value] of Object.entries(values)) substitute(document, `result.${name}`, value);
      expect(() => parseCandidateJob(document)).toThrow("scientific result");
    }
  });
});

describe("candidate observation identity", () => {
  it("accepts a real state transition while keeping immutable custody", () => {
    const original = parseCandidateJob({ ...specimen(), status: "running", result: null });
    const terminal = parseCandidateJob(specimen());
    expect(bindCandidateJobObservation(original, terminal)).toBe(terminal);
  });

  it.each([
    ["job_id", "sj_fedcba9876543210"], ["owner", "bob"],
    ["created_at_utc", "2026-10-29T00:00:00Z"], ["request_id", "other-request"],
    ["admission.candidate_operation", "review"], ["admission.candidate_name", "ChangedCandidate"],
    ["admission.candidate_sha256", REQUEST], ["admission.request_sha256", HASH],
    ["admission.projected_steps", 3],
  ])("rejects a valid but relabelled observation: %s", (path, replacement) => {
    const value: Record<string, unknown> = { ...specimen(), status: "running", result: null };
    const original = parseCandidateJob(value);
    const changed = structuredClone(value);
    substitute(changed, path, replacement);
    if (path === "admission.request_sha256") changed.experiment_sha256 = replacement;
    // The substituted record passes its independent grammar. Only the binding
    // to the original admission can establish that it is the same calculation.
    const observed = parseCandidateJob(changed);
    expect(() => bindCandidateJobObservation(original, observed)).toThrow("admitted input identity");
  });
});
