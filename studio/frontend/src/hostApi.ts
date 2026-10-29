// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — bounded parent-only scientific runtime client

import { HOST_API_PROTOCOL, HOST_OPERATIONS, type HostApiRequest, type HostArtifact,
  type HostJob, type HostJson, type HostOperation } from "./hostApiTypes";

/** Authored client refusal that never contains backend diagnostics. */
export class HostApiRefusal extends Error {}
const MAX_BYTES = 8 * 1024 * 1024;
const DIGEST = /^sha256:[0-9a-f]{64}$/;
const JOB = /^sj_[0-9a-f]{16}$/;
const STATUSES = new Set(["pending", "running", "completed", "failed", "cancelling", "cancelled",
  "timed_out", "interrupted", "unknown"]);

/**
 * Narrow structured-clone data without accepting arrays as records.
 *
 * @param value - Untrusted bridge response.
 * @returns Whether value is a non-array object.
 */
function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Send only the closed compute capability to the actual immediate parent.
 *
 * @param request - Typed fixed action, with original scientific JSON when submitting.
 * @param signal - Actual mounted panel lifetime.
 * @returns Correlated response data; listeners and timers are removed on every exit.
 */
function exchange(request: HostApiRequest, signal: AbortSignal): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) { reject(new HostApiRefusal("Runtime request cancelled.")); return; }
    const id = crypto.randomUUID(), parent = window.parent;
    const timer = window.setTimeout(() => {
      cleanup(); reject(new HostApiRefusal("Runtime request timed out."));
    }, 45_000);
    /** Release callbacks owned exclusively by this request. */
    function cleanup(): void {
      window.clearTimeout(timer); window.removeEventListener("message", receive);
      signal.removeEventListener("abort", cancel);
    }
    /** Reject an unmounted caller before accepting any late response. */
    function cancel(): void { cleanup(); reject(new HostApiRefusal("Runtime request cancelled.")); }
    /**
     * Validate browser-assigned source and correlation before trusting response data.
     *
     * @param event - Actual window message.
     */
    function receive(event: MessageEvent<unknown>): void {
      const data = event.data;
      if (event.source !== parent || !record(data) || data.type !== "scpn:bridge:response"
        || data.id !== id) return;
      cleanup();
      if (data.ok !== true || !record(data.result) || data.result.protocol !== HOST_API_PROTOCOL) {
        reject(new HostApiRefusal("Local runtime request was refused."));
      } else resolve(data.result);
    }
    window.addEventListener("message", receive);
    signal.addEventListener("abort", cancel, { once: true });
    try {
      parent.postMessage({ type: "scpn:bridge:request", id,
        capability: "custom:sc-neurocore:api", payload: { protocol: HOST_API_PROTOCOL, ...request } }, "*");
    } catch {
      cleanup(); reject(new HostApiRefusal("Runtime request could not be sent."));
    }
  });
}

/**
 * Discover operation families through the actual configured host service.
 *
 * @param signal - Panel lifetime.
 * @returns Unique known operations reported by the installed backend.
 */
export async function describeHost(signal: AbortSignal): Promise<HostOperation[]> {
  const data = await exchange({ action: "describe" }, signal);
  if (!Array.isArray(data.operations) || !data.operations.every((item: unknown): item is HostOperation =>
    typeof item === "string" && HOST_OPERATIONS.some((known) => known === item))
    || new Set(data.operations).size !== data.operations.length) {
    throw new HostApiRefusal("Runtime operation catalogue is invalid.");
  }
  return data.operations;
}

/**
 * Validate the finite path-free job projection returned by the host.
 *
 * @param data - Correlated host response.
 * @returns Checked job record.
 */
function job(data: Record<string, unknown>): HostJob {
  if (typeof data.jobId !== "string" || JOB.exec(data.jobId)?.[0] !== data.jobId
    || (data.kind !== "analysis" && data.kind !== "compiler")
    || typeof data.status !== "string" || !STATUSES.has(data.status)
    || !Array.isArray(data.artifacts) || data.artifacts.length > 1024
    || !data.artifacts.every((entry: unknown): entry is HostArtifact => record(entry)
      && typeof entry.digest === "string" && DIGEST.exec(entry.digest)?.[0] === entry.digest
      && typeof entry.bytes === "number" && Number.isSafeInteger(entry.bytes) && entry.bytes >= 0
      && typeof entry.result === "boolean")) throw new HostApiRefusal("Runtime job response is invalid.");
  return { jobId: data.jobId, kind: data.kind, status: data.status, artifacts: data.artifacts };
}

/**
 * Submit original scientific JSON without reserializing numeric tokens or object order.
 * Shape validation never reads or rewrites the decoded numeric values.
 *
 * @param operation - Closed scientific action.
 * @param json - Original object text from the editor.
 * @param signal - Panel lifetime.
 * @returns The actual admitted job, after validating its expected family.
 */
export async function submitHostJob(operation: HostOperation, json: string, signal: AbortSignal): Promise<HostJob> {
  const parsed: unknown = JSON.parse(json);
  if (!record(parsed) || new TextEncoder().encode(json).length > 1_048_576) {
    throw new HostApiRefusal("Enter a scientific request object within the one MiB limit.");
  }
  const admitted = job(await exchange({ action: "submit", operation, json }, signal));
  if (admitted.kind !== (operation.startsWith("ir.") ? "compiler" : "analysis")) {
    throw new HostApiRefusal("Runtime job family does not match the request.");
  }
  return admitted;
}

/**
 * Inspect or cancel exactly one admitted job without exposing backend routes.
 *
 * @param jobId - Panel-admitted job identifier.
 * @param action - Read state or request actual cancellation.
 * @param signal - Panel lifetime.
 * @returns Checked current record for the same job.
 */
export async function readHostJob(jobId: string, action: "status" | "cancel", signal: AbortSignal): Promise<HostJob> {
  const current = job(await exchange({ action, jobId }, signal));
  if (current.jobId !== jobId) throw new HostApiRefusal("Runtime answered for another job.");
  return current;
}

/**
 * Verify exact UTF-8 bytes against the host's independently checked digest.
 *
 * @param bytes - Complete finite artifact.
 * @param digest - SHA-256 content identity.
 * @param size - Declared measured byte count.
 * @param limit - Result/artifact ceiling, or the separate catalogue response ceiling.
 * @returns Completion after matching size and SHA-256, otherwise a fixed refusal.
 */
async function integrity(bytes: Uint8Array<ArrayBuffer>, digest: unknown, size: unknown, limit = MAX_BYTES): Promise<void> {
  if (typeof digest !== "string" || DIGEST.exec(digest)?.[0] !== digest || size !== bytes.length || bytes.length > limit) {
    throw new HostApiRefusal("Runtime artifact size or digest is invalid.");
  }
  const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  if (`sha256:${Array.from(hash, (item) => item.toString(16).padStart(2, "0")).join("")}` !== digest) {
    throw new HostApiRefusal("Runtime artifact integrity check failed.");
  }
}

/**
 * Bind an accepted textual compiler field to its actual declared artifact.
 *
 * @param text - Exact result or graph text, with no reserialization.
 * @returns SHA-256 of its UTF-8 bytes.
 */
export async function hostedTextDigest(text: string): Promise<string> {
  const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)));
  return `sha256:${Array.from(hash, (item) => item.toString(16).padStart(2, "0")).join("")}`;
}

/**
 * Read genuine model/template JSON or a completed job's sealed scientific result.
 *
 * @param request - Fixed catalogue selection or panel-admitted result identity.
 * @param signal - Panel lifetime.
 * @returns Original JSON text with its verified content binding.
 */
export async function readHostJson(request: { action: "models" | "templates" } | { action: "result"; jobId: string },
  signal: AbortSignal): Promise<HostJson> {
  const data = await exchange(request, signal);
  const limit = request.action === "result" ? MAX_BYTES : 16 * 1024 * 1024;
  if (typeof data.json !== "string" || data.json.length > limit
    || typeof data.digest !== "string" || typeof data.bytes !== "number") throw new HostApiRefusal("Runtime JSON response is invalid.");
  const bytes = new TextEncoder().encode(data.json);
  await integrity(bytes, data.digest, data.bytes, limit);
  JSON.parse(data.json);
  return { json: data.json, digest: data.digest, bytes: bytes.length };
}

/**
 * Download only a declared digest, verifying its finite bytes independently.
 *
 * @param jobId - Panel-admitted job.
 * @param digest - Digest from that job's actual artifact list.
 * @param signal - Panel lifetime.
 * @returns Exact bytes suitable for a user-requested Blob download.
 */
export async function readHostArtifact(jobId: string, digest: string, signal: AbortSignal): Promise<Uint8Array<ArrayBuffer>> {
  const data = await exchange({ action: "artifact", jobId, digest }, signal);
  if (data.encoding !== "base64" || typeof data.data !== "string" || data.data.length > 11_184_812
    || data.digest !== digest) throw new HostApiRefusal("Runtime artifact response is invalid.");
  let bytes: Uint8Array<ArrayBuffer>;
  try { bytes = Uint8Array.from(atob(data.data), (item) => item.charCodeAt(0)); }
  catch { throw new HostApiRefusal("Runtime artifact encoding is invalid."); }
  await integrity(bytes, digest, data.bytes);
  return bytes;
}

/**
 * Ask the trusted parent to download a declared artifact without a sandbox grant.
 *
 * @param jobId - Panel-admitted job identity.
 * @param artifact - Actual job artifact selected by digest, without a caller filename.
 * @param signal - Mounted panel lifetime.
 * @returns Completion after matching the parent's dispatched-download receipt.
 */
export async function downloadHostArtifact(jobId: string, artifact: HostArtifact, signal: AbortSignal): Promise<void> {
  const data = await exchange({ action: "download", jobId, digest: artifact.digest }, signal);
  if (data.downloaded !== true || data.digest !== artifact.digest || data.bytes !== artifact.bytes) {
    throw new HostApiRefusal("Runtime download receipt is invalid.");
  }
}
