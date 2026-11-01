// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Actor-scoped durable graph HTTP client

import { authHeaders, json } from "./http";
import type { NetworkGraph } from "./types";
import { sealSha256 } from "../evidenceSeal";
import { snapshotGraphJobRequest, type GraphJobRequestSnapshot } from "../graphJobRequest";
import {
  parseGraphJob, parseGraphJobHistory, parseGraphJobReceipt, parseGraphJobValidation,
  type GraphJob, type GraphJobReceipt, type GraphJobValidation,
} from "../graphJobs";
import { parseStudioJson } from "../studioJson";
import { readGraphJobArtifact, type GraphJobArtifact } from "../graphJobResultTransport";

/** Immutable wire snapshot and the independently checked server interpretation. */
export interface PreparedGraphJob {
  request: GraphJobRequestSnapshot;
  identity: GraphJobValidation;
}

/**
 * Construct a fixed graph-job route from one admissible identifier.
 *
 * @param jobId - Selected ledger identifier, including when restored from storage.
 * @returns Its same-origin status route; no response-supplied URL is followed.
 */
function jobRoute(jobId: string): string {
  if (!/^sj_[0-9a-f]{16}$/.test(jobId)) throw new Error("Invalid graph job identifier.");
  return `/api/graph/jobs/${jobId}`;
}

/**
 * Resolve a graph's actual async identity without allocating or running a job.
 *
 * @param graph - Current scientific inputs and layout to capture before awaiting.
 * @param signal - Caller context cancellation, including account/project switches.
 * @returns Exact captured JSON and a request-bound server validation.
 */
export async function prepareGraphJob(graph: NetworkGraph, signal?: AbortSignal): Promise<PreparedGraphJob> {
  const request = snapshotGraphJobRequest(graph);
  const response = await fetch("/api/graph/jobs/validate", {
    method: "POST", headers: { "Content-Type": "application/json", ...authHeaders() },
    body: request.json, signal,
  });
  const identity = await parseGraphJobValidation(await json<unknown>(response), parseStudioJson(request.json));
  return { request, identity };
}

/**
 * Submit or recover the same captured request under the same durable retry key.
 *
 * A transport failure may follow a real admission. The caller retains this
 * snapshot and key instead of silently creating another calculation.
 *
 * @param prepared - Previously checked input snapshot, never a live mutable graph.
 * @param owner - Authenticated actor retained before this operation.
 * @param retryKey - Original visible-ASCII idempotency key, reused for recovery.
 * @param signal - Context cancellation checked before the allocating request.
 * @returns Checked admission acknowledgement; its job may still be pending.
 */
export async function submitGraphJob(
  prepared: PreparedGraphJob, owner: string, retryKey: string, signal?: AbortSignal,
): Promise<GraphJobReceipt> {
  if (!/^[!-~]{1,128}$/.test(retryKey)) throw new Error("Invalid graph submission retry key.");
  const requestJson = prepared.request.json;
  const identity = { ...prepared.identity };
  const headers = authHeaders();
  if (await sealSha256(parseStudioJson(requestJson)) !== identity.request_seal_sha256) {
    throw new Error("Graph submission no longer matches its prepared input.");
  }
  signal?.throwIfAborted();
  if (authHeaders().Authorization !== headers.Authorization) {
    throw new Error("Studio account changed before graph submission.");
  }
  const response = await fetch("/api/graph/jobs", {
    method: "POST", headers: {
      "Content-Type": "application/json", ...headers, "Idempotency-Key": retryKey,
      "X-Studio-Graph-Submission-SHA256": identity.submission_sha256,
    },
    body: requestJson, signal,
  });
  return parseGraphJobReceipt(await json<unknown>(response), owner, identity);
}

/**
 * Observe a particular actor-owned graph calculation without resubmitting it.
 *
 * @param jobId - Previously selected ledger identifier.
 * @param owner - Current authenticated actor.
 * @param signal - Caller context cancellation.
 * @returns Independently parsed and digest-bound status.
 */
export async function getGraphJob(jobId: string, owner: string, signal?: AbortSignal): Promise<GraphJob> {
  const response = await fetch(jobRoute(jobId), { headers: authHeaders(), signal });
  return parseGraphJob(await json<unknown>(response), owner, jobId);
}

/**
 * Recover bounded actor-owned graph history from the durable server ledger.
 *
 * @param owner - Current authenticated actor, never the stored previous login.
 * @param signal - Caller context cancellation.
 * @returns Newest-first checked summaries; omitted results stay omitted.
 */
export async function listGraphJobs(owner: string, signal?: AbortSignal): Promise<GraphJob[]> {
  const response = await fetch("/api/graph/jobs?limit=100", { headers: authHeaders(), signal });
  return parseGraphJobHistory(await json<unknown>(response), owner);
}

/**
 * Ask the authority to stop this job without pretending cancellation is immediate.
 *
 * @param jobId - Selected graph calculation.
 * @param owner - Current authenticated actor.
 * @param signal - Caller context cancellation.
 * @returns Actual checked supervisor state, including cancelling or already completed.
 */
export async function cancelGraphJob(jobId: string, owner: string, signal?: AbortSignal): Promise<GraphJob> {
  const response = await fetch(jobRoute(jobId) + "/cancel", { method: "POST", headers: authHeaders(), signal });
  return parseGraphJob(await json<unknown>(response), owner, jobId);
}

/**
 * Download a selected completed job's entire verified scientific result.
 *
 * @param job - Checked actor-owned completed status, never a history summary.
 * @param signal - Context cancellation while downloading or switching accounts.
 * @returns Exact verified bytes and scientific result; graph currency is checked by the caller before adoption.
 */
export async function downloadGraphJobArtifact(job: GraphJob, signal?: AbortSignal): Promise<GraphJobArtifact> {
  const captured = { ...job, admission: { ...job.admission }, result: job.result === null ? null : { ...job.result } };
  if (captured.status !== "completed" || captured.result === null) throw new Error("Graph job has no complete result.");
  const response = await fetch(jobRoute(captured.job_id) + "/result", { headers: authHeaders(), signal });
  if (!response.ok) await json<unknown>(response);
  return readGraphJobArtifact(response, captured);
}
