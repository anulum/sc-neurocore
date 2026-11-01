// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Real HTTP graph-contract admission and tamper controls

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  bindGraphJobObservation, bindGraphJobPreparation, graphJobActive, parseGraphJob, parseGraphJobHistory,
  parseGraphJobReceipt, parseGraphJobValidation,
} from "./graphJobs";

/** Read one untrusted object from the recorded real HTTP/worker contract. */
function object(value: unknown): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) throw new Error("Fixture is not an object");
  return value as Record<string, unknown>;
}

/** Keep each case independent while retaining the actual producer's payload shape. */
function fixture(): Record<string, unknown> {
  const value: unknown = JSON.parse(readFileSync(new URL("./test-data/graphJobContract.json", import.meta.url), "utf8"));
  return object(value);
}

describe("server-resolved graph validation", () => {
  it("binds a recovered real job to its retained preparation and refuses another scientific submission", async () => {
    const sample = fixture();
    const job = await parseGraphJob(sample.completed, "alice");
    const prepared = await parseGraphJobValidation(sample.validation, sample.request);
    expect(bindGraphJobPreparation(job, prepared)).toBe(job);
    for (const field of ["graph_sha256", "request_sha256", "submission_sha256", "projected_neuron_steps"] as const) {
      const changed = { ...prepared, [field]: field === "projected_neuron_steps" ? prepared.projected_neuron_steps + 1 : "0".repeat(64) };
      expect(() => bindGraphJobPreparation(job, changed)).toThrow("changed the retained submission");
    }
  });
  it("independently verifies the real Python request seal and mode-bound submission digest", async () => {
    const sample = fixture();
    const checked = await parseGraphJobValidation(sample.validation, sample.request);
    expect(checked.projected_neuron_steps).toBe(800);
    expect(checked.n_steps * checked.n_neurons).toBe(checked.projected_neuron_steps);
    const admitted = await parseGraphJobReceipt(sample.admission, "alice", checked);
    expect(admitted.job.admission.graph_sha256).toBe(checked.graph_sha256);
    expect(admitted.job.experiment_sha256).toBe(checked.submission_sha256);
  });

  it.each([
    ["schema_version", "unknown"], ["execution_mode", "synchronous"],
    ["graph_sha256", "bad"], ["request_sha256", null], ["request_seal_sha256", "bad"],
    ["submission_sha256", "bad"], ["n_steps", 0], ["n_neurons", 0],
    ["projected_neuron_steps", -1], ["projected_neuron_steps", 1], ["n_steps", Infinity],
    ["n_neurons", "4"],
  ])("refuses malformed or inconsistent preview field %s", async (field, value) => {
    const sample = fixture();
    await expect(parseGraphJobValidation({ ...object(sample.validation), [field]: value }, sample.request)).rejects.toThrow("Invalid asynchronous graph validation");
  });

  it("refuses a different exact request and a correctly shaped but wrong submission hash", async () => {
    const sample = fixture();
    await expect(parseGraphJobValidation(sample.validation, { ...object(sample.request), seed: 43 })).rejects.toThrow("does not match");
    await expect(parseGraphJobValidation({ ...object(sample.validation), submission_sha256: "0".repeat(64) }, sample.request)).rejects.toThrow("does not match");
    await expect(parseGraphJobValidation(null, sample.request)).rejects.toThrow("Invalid asynchronous");
  });
});

describe("actor-owned graph records", () => {
  it("reads actual pending acknowledgement, completed custody and result-free history", async () => {
    const sample = fixture();
    const completed = await parseGraphJob(sample.completed, "alice");
    expect(completed.result?.result_size_bytes).toBe(9866);
    expect(completed.result?.graph_sha256).toBe(completed.admission.graph_sha256);
    const history = await parseGraphJobHistory(sample.history, "alice");
    expect(history.map((job) => job.job_id)).toEqual([completed.job_id]);
    expect(history[0]?.result).toBeNull();
    const pending = await parseGraphJob(object(sample.admission).job, "alice");
    expect(pending.owner).toBe("alice");
    expect(bindGraphJobObservation(pending, completed)).toBe(completed);
  });

  it.each([
    ["job_id", "not-a-job"], ["owner", "bob"], ["kind", "compiler"],
    ["execution_model", "thread"], ["workspace", ""], ["workspace", null],
    ["experiment_sha256", "bad"],
  ])("refuses foreign or malformed record field %s", async (field, value) => {
    const sample = fixture();
    await expect(parseGraphJob({ ...object(sample.completed), [field]: value }, "alice")).rejects.toThrow("Invalid graph job record or actor");
  });

  it("binds the selected ID, actual actor and complete submission digest", async () => {
    const sample = fixture();
    const record = object(sample.completed);
    await expect(parseGraphJob(null, "alice")).rejects.toThrow("Invalid graph job record");
    await expect(parseGraphJob(record, "")).rejects.toThrow("Invalid graph job record");
    await expect(parseGraphJob(record, "alice", "sj_0000000000000000")).rejects.toThrow("Invalid graph job record");
    await expect(parseGraphJob({ ...record, experiment_sha256: "0".repeat(64) }, "alice")).rejects.toThrow("digest is inconsistent");
  });

  it.each([
    ["graph_operation", "fit"], ["graph_sha256", "bad"], ["request_sha256", "bad"],
    ["projected_neuron_steps", 0], ["projected_neuron_steps", NaN],
  ])("refuses graph input identity field %s", async (field, value) => {
    const record = object(fixture().completed);
    await expect(parseGraphJob({ ...record, admission: { ...object(record.admission), [field]: value } }, "alice")).rejects.toThrow("Invalid graph job input identity");
  });

  it("does not repair absent admission metadata", async () => {
    const record = object(fixture().completed);
    await expect(parseGraphJob({ ...record, admission: null }, "alice")).rejects.toThrow("Invalid graph job input identity");
  });

  it.each([
    ["schema_version", "unknown"], ["graph_sha256", "0".repeat(64)],
    ["result_artifact", "../other.json"], ["result_sha256", "bad"], ["result_size_bytes", 0],
    ["n_steps", 0], ["n_total", 0], ["n_spikes", -1], ["n_steps", 201],
  ])("refuses compact result field %s", async (field, value) => {
    const record = object(fixture().completed);
    await expect(parseGraphJob({ ...record, result: { ...object(record.result), [field]: value } }, "alice")).rejects.toThrow("Invalid graph result receipt");
  });

  it("distinguishes an intentionally omitted history result from an invalid terminal result", async () => {
    const record = object(fixture().completed);
    await expect(parseGraphJob({ ...record, result: null }, "alice")).rejects.toThrow("state does not permit");
    await expect(parseGraphJob({ ...record, status: "failed" }, "alice")).rejects.toThrow("state does not permit");
    await expect(parseGraphJob({ ...record, result: null }, "alice", undefined, true)).resolves.toHaveProperty("result", null);
  });

  it.each(["absent", "duplicate", "sha", "size"])("refuses an artifact declaration that is %s", async (fault) => {
    const record = object(fixture().completed);
    const artifact: unknown = Array.isArray(record.artifacts) ? record.artifacts[0] : null;
    const item = object(artifact);
    const artifacts = fault === "absent" ? [] : fault === "duplicate" ? [item, item]
      : [{ ...item, ...(fault === "sha" ? { sha256: "0".repeat(64) } : { size_bytes: 1 }) }];
    await expect(parseGraphJob({ ...record, artifacts }, "alice")).rejects.toThrow("declared artifact");
  });
});

describe("acknowledgement, observation and history custody", () => {
  it.each(["job_id", "status_route", "graph_sha256", "request_sha256", "projected_neuron_steps"])("rejects a changed acknowledgement %s", async (field) => {
    const sample = fixture();
    const expected = await parseGraphJobValidation(sample.validation, sample.request);
    await expect(parseGraphJobReceipt({ ...object(sample.admission), [field]: "other" }, "alice", expected)).rejects.toThrow("changed the prepared input");
  });

  it("refuses unsupported acknowledgement versions and a different prepared calculation", async () => {
    const sample = fixture();
    const expected = await parseGraphJobValidation(sample.validation, sample.request);
    await expect(parseGraphJobReceipt(null, "alice", expected)).rejects.toThrow("Invalid graph job acknowledgement");
    await expect(parseGraphJobReceipt({ ...object(sample.admission), execution_mode: "sync" }, "alice", expected)).rejects.toThrow("Invalid graph job acknowledgement");
    await expect(parseGraphJobReceipt(sample.admission, "alice", { ...expected, submission_sha256: "0".repeat(64) })).rejects.toThrow("changed the prepared input");
  });

  it("rejects immutable custody changes between individually readable observations", async () => {
    const job = await parseGraphJob(fixture().completed, "alice");
    for (const observed of [
      { ...job, job_id: "sj_0000000000000000" }, { ...job, owner: "bob" }, { ...job, workspace: "other" },
      { ...job, created_at_utc: "other" }, { ...job, request_id: "other" }, { ...job, experiment_sha256: "0".repeat(64) },
      { ...job, admission: { ...job.admission, graph_sha256: "0".repeat(64) } },
      { ...job, admission: { ...job.admission, request_sha256: "0".repeat(64) } },
      { ...job, admission: { ...job.admission, projected_neuron_steps: 801 } },
    ]) expect(() => bindGraphJobObservation(job, observed)).toThrow("changed the selected calculation");
    expect(graphJobActive({ ...job, status: "pending" })).toBe(true);
    expect(graphJobActive({ ...job, status: "running" })).toBe(true);
    expect(graphJobActive({ ...job, status: "cancelling" })).toBe(true);
    expect(graphJobActive(job)).toBe(false);
  });

  it("refuses unsupported, oversized, duplicate and other-actor history", async () => {
    const sample = fixture();
    const history = object(sample.history);
    const jobs: unknown[] = Array.isArray(history.jobs) ? history.jobs : [];
    await expect(parseGraphJobHistory(null, "alice")).rejects.toThrow("Invalid graph job history");
    await expect(parseGraphJobHistory({ ...history, schema_version: "other" }, "alice")).rejects.toThrow("Invalid graph job history");
    await expect(parseGraphJobHistory({ ...history, jobs: "bad" }, "alice")).rejects.toThrow("Invalid graph job history");
    await expect(parseGraphJobHistory({ ...history, jobs: Array.from({ length: 101 }, () => jobs[0]) }, "alice")).rejects.toThrow("Invalid graph job history");
    await expect(parseGraphJobHistory({ ...history, jobs: [jobs[0], jobs[0]] }, "alice")).rejects.toThrow("repeats a job");
    await expect(parseGraphJobHistory(history, "bob")).rejects.toThrow("Invalid graph job record or actor");
    await expect(parseGraphJobHistory({ ...history, jobs: [] }, "alice")).resolves.toEqual([]);
  });
});
