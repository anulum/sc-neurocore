// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Graph job identity and complete-result receipts

import { parseStudioJobRecord } from "./analysisJobRecordValidation";
import type { StudioJobRecord } from "./api/types";
import { sealSha256 } from "./evidenceSeal";

/** Server-resolved asynchronous identity of one exact, independently sealed request. */
export interface GraphJobValidation {
  graph_sha256: string;
  request_sha256: string;
  request_seal_sha256: string;
  submission_sha256: string;
  projected_neuron_steps: number;
  n_steps: number;
  n_neurons: number;
}

/** Scientific input custody retained by the durable authority. */
export interface GraphJobAdmission {
  graph_operation: "simulate";
  graph_sha256: string;
  request_sha256: string;
  projected_neuron_steps: number;
}

/** Compact custody receipt; scientific arrays live in the complete artifact. */
export interface GraphJobResultReceipt {
  schema_version: "studio.graph.job.result.v1";
  graph_sha256: string;
  result_artifact: "graph/result.json";
  result_sha256: string;
  result_size_bytes: number;
  n_steps: number;
  n_total: number;
  n_spikes: number;
}

/** Actor- and workspace-bound graph record with a checked artifact receipt. */
export interface GraphJob extends Omit<StudioJobRecord, "result"> {
  workspace: string;
  experiment_sha256: string;
  admission: GraphJobAdmission;
  result: GraphJobResultReceipt | null;
}

/** Admission acknowledgement; a pending job is not a completed calculation. */
export interface GraphJobReceipt {
  job: GraphJob;
  status_route: string;
}

/**
 * Read a JSON object's shape without accepting arrays or null.
 *
 * @param value - Untrusted JSON value.
 * @returns Whether named fields can be inspected.
 */
function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/**
 * Check the wire representation of a lower-case SHA-256 digest.
 *
 * @param value - Untrusted field.
 * @returns Whether it is a digest representation, without asserting its content.
 */
function digest(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
}

/**
 * Check an exactly representable nonnegative count.
 *
 * @param value - Untrusted field.
 * @returns Whether arithmetic can use this count without rounding an integer.
 */
function count(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

/**
 * Hash complete bytes, preserving their encoding rather than reserializing JSON.
 *
 * @param bytes - Caller-owned bytes to verify.
 * @returns Lower-case SHA-256; unavailable Web Crypto rejects the operation.
 */
export async function graphBytesSha256(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  const hashed = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(hashed), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

/**
 * Verify the server preview against the exact request sent by this caller.
 *
 * The maintained cross-runtime seal handles Python/JavaScript number rendering.
 * The separate raw seed checksum and mode-bound graph digest remain unchanged.
 *
 * @param value - Untrusted nonallocating validation response.
 * @param request - Exact JSON-shaped wire request retained before the HTTP call.
 * @returns A copied, checked identity; no worker or admission is implied.
 */
export async function parseGraphJobValidation(value: unknown, request: unknown): Promise<GraphJobValidation> {
  if (!record(value) || value.schema_version !== "studio.graph.job.validation.v1"
    || value.execution_mode !== "async_job" || !digest(value.graph_sha256)
    || !digest(value.request_sha256) || !digest(value.request_seal_sha256)
    || !digest(value.submission_sha256) || !count(value.projected_neuron_steps)
    || !count(value.n_steps) || value.n_steps === 0 || !count(value.n_neurons) || value.n_neurons === 0
    || value.projected_neuron_steps !== value.n_steps * value.n_neurons) {
    throw new Error("Invalid asynchronous graph validation.");
  }
  const checked: GraphJobValidation = {
    graph_sha256: value.graph_sha256, request_sha256: value.request_sha256,
    request_seal_sha256: value.request_seal_sha256, submission_sha256: value.submission_sha256,
    projected_neuron_steps: value.projected_neuron_steps, n_steps: value.n_steps, n_neurons: value.n_neurons,
  };
  if (await sealSha256(request) !== checked.request_seal_sha256
    || await graphBytesSha256(new TextEncoder().encode(`${checked.request_sha256}:${checked.graph_sha256}`)) !== checked.submission_sha256) {
    throw new Error("Graph validation does not match the submitted request.");
  }
  return checked;
}

/**
 * Read the compact receipt and bind every count and artifact identity.
 *
 * @param value - Untrusted completed job result.
 * @param admission - Already checked scientific input custody.
 * @returns A copied receipt with no unverified scientific arrays.
 */
function resultReceipt(value: unknown, admission: GraphJobAdmission): GraphJobResultReceipt {
  if (!record(value) || value.schema_version !== "studio.graph.job.result.v1"
    || value.graph_sha256 !== admission.graph_sha256 || value.result_artifact !== "graph/result.json"
    || !digest(value.result_sha256) || !count(value.result_size_bytes) || value.result_size_bytes === 0
    || !count(value.n_steps) || value.n_steps === 0 || !count(value.n_total) || value.n_total === 0
    || !count(value.n_spikes) || value.n_steps * value.n_total !== admission.projected_neuron_steps) {
    throw new Error("Invalid graph result receipt.");
  }
  return {
    schema_version: "studio.graph.job.result.v1", graph_sha256: admission.graph_sha256,
    result_artifact: "graph/result.json", result_sha256: value.result_sha256,
    result_size_bytes: value.result_size_bytes, n_steps: value.n_steps, n_total: value.n_total,
    n_spikes: value.n_spikes,
  };
}

/**
 * Admit an actor-owned ledger observation before presenting it as a graph job.
 *
 * @param value - Untrusted status, submission or history record.
 * @param owner - Current authenticated actor, never a value from the response.
 * @param expectedId - Selected job ID when reading one job.
 * @param summary - History explicitly omits the compact result too.
 * @returns A copied record with verified mode-bound submission identity.
 */
export async function parseGraphJob(value: unknown, owner: string, expectedId?: string, summary = false): Promise<GraphJob> {
  const base = parseStudioJobRecord(value, { expectedKind: "analysis", expectedJobId: expectedId });
  if (!base.ok || !record(value) || !/^sj_[0-9a-f]{16}$/.test(base.value.job_id)
    || owner.length === 0 || base.value.owner !== owner || base.value.execution_model !== "process"
    || typeof value.workspace !== "string" || value.workspace.length === 0 || !digest(value.experiment_sha256)) {
    throw new Error("Invalid graph job record or actor.");
  }
  const raw = value.admission;
  if (!record(raw) || raw.graph_operation !== "simulate" || !digest(raw.graph_sha256)
    || !digest(raw.request_sha256) || !count(raw.projected_neuron_steps) || raw.projected_neuron_steps === 0) {
    throw new Error("Invalid graph job input identity.");
  }
  const admission: GraphJobAdmission = {
    graph_operation: "simulate", graph_sha256: raw.graph_sha256,
    request_sha256: raw.request_sha256, projected_neuron_steps: raw.projected_neuron_steps,
  };
  const result = base.value.result === null ? null : resultReceipt(base.value.result, admission);
  if (base.value.status !== "completed" && result !== null
    || !summary && base.value.status === "completed" && result === null) {
    throw new Error("Graph job state does not permit this result.");
  }
  const declared = base.value.artifacts.filter((artifact) => artifact.relative_path === "graph/result.json");
  if (result !== null && (declared.length !== 1 || declared[0]?.sha256 !== result.result_sha256
    || declared[0].size_bytes !== result.result_size_bytes)) {
    throw new Error("Graph result receipt does not match its declared artifact.");
  }
  const job: GraphJob = { ...base.value, workspace: value.workspace,
    experiment_sha256: value.experiment_sha256, admission, result };
  if (await graphBytesSha256(new TextEncoder().encode(`${admission.request_sha256}:${admission.graph_sha256}`)) !== job.experiment_sha256) {
    throw new Error("Graph job submission digest is inconsistent.");
  }
  return job;
}

/**
 * Bind an admission response to the independently checked input preview.
 *
 * @param value - Untrusted submit response.
 * @param owner - Authenticated submitter retained by the caller.
 * @param expected - Exact prepared input identity.
 * @returns A checked acknowledgement suitable for observation and cancellation.
 */
export async function parseGraphJobReceipt(value: unknown, owner: string, expected: GraphJobValidation): Promise<GraphJobReceipt> {
  if (!record(value) || value.schema_version !== "studio.graph.job.v1" || value.execution_mode !== "async_job") {
    throw new Error("Invalid graph job acknowledgement.");
  }
  const job = await parseGraphJob(value.job, owner);
  const route = `/api/graph/jobs/${job.job_id}`;
  if (value.job_id !== job.job_id || value.status_route !== route
    || value.graph_sha256 !== expected.graph_sha256 || job.admission.graph_sha256 !== expected.graph_sha256
    || value.request_sha256 !== expected.request_sha256 || job.admission.request_sha256 !== expected.request_sha256
    || value.projected_neuron_steps !== expected.projected_neuron_steps
    || job.admission.projected_neuron_steps !== expected.projected_neuron_steps
    || job.experiment_sha256 !== expected.submission_sha256) {
    throw new Error("Graph acknowledgement changed the prepared input identity.");
  }
  return { job, status_route: route };
}

/**
 * Bind a recovered selected job to its retained independently checked preparation.
 *
 * @param job - Actor-owned current server observation.
 * @param expected - Checked original submission interpretation.
 * @returns The same observation after scientific custody comparisons.
 */
export function bindGraphJobPreparation(job: GraphJob, expected: GraphJobValidation): GraphJob {
  if (job.admission.graph_sha256 !== expected.graph_sha256 || job.admission.request_sha256 !== expected.request_sha256
    || job.admission.projected_neuron_steps !== expected.projected_neuron_steps || job.experiment_sha256 !== expected.submission_sha256) {
    throw new Error("Recovered graph job changed the retained submission.");
  }
  return job;
}

/**
 * Keep successive observations bound to the originally selected calculation.
 *
 * @param expected - Original checked acknowledgement or history selection.
 * @param observed - Independently checked poll or cancellation response.
 * @returns The observation after immutable custody comparisons.
 */
export function bindGraphJobObservation(expected: GraphJob, observed: GraphJob): GraphJob {
  if (expected.job_id !== observed.job_id || expected.owner !== observed.owner
    || expected.workspace !== observed.workspace || expected.created_at_utc !== observed.created_at_utc
    || expected.request_id !== observed.request_id || expected.experiment_sha256 !== observed.experiment_sha256
    || expected.admission.graph_sha256 !== observed.admission.graph_sha256
    || expected.admission.request_sha256 !== observed.admission.request_sha256
    || expected.admission.projected_neuron_steps !== observed.admission.projected_neuron_steps) {
    throw new Error("Graph observation changed the selected calculation.");
  }
  return observed;
}

/**
 * Read bounded actor-owned history without inventing omitted result data.
 *
 * @param value - Untrusted newest-first history response.
 * @param owner - Current authenticated actor.
 * @returns Checked records in the server's original order.
 */
export async function parseGraphJobHistory(value: unknown, owner: string): Promise<GraphJob[]> {
  if (!record(value) || value.schema_version !== "studio.graph.jobs.v1" || !Array.isArray(value.jobs)
    || value.jobs.length > 100) throw new Error("Invalid graph job history.");
  const jobs = await Promise.all(value.jobs.map((item: unknown) => parseGraphJob(item, owner, undefined, true)));
  if (new Set(jobs.map((job) => job.job_id)).size !== jobs.length) throw new Error("Graph history repeats a job.");
  return jobs;
}

/**
 * Decide whether another observation can report ordinary running progress.
 *
 * @param job - Checked current observation.
 * @returns Whether it is pending, running or awaiting cancellation.
 */
export function graphJobActive(job: GraphJob): boolean {
  return ["pending", "running", "cancelling"].includes(job.status);
}
