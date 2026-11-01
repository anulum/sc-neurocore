// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Bounded graph result response transport

import { GRAPH_RESULT_MAX_BYTES, parseGraphJobResult, type GraphJobSimulationResult } from "./graphJobResult";
import type { GraphJob } from "./graphJobs";

/** Exact verified export bytes alongside the checked scientific presentation data. */
export interface GraphJobArtifact {
  bytes: Uint8Array<ArrayBuffer>;
  result: GraphJobSimulationResult;
}

/**
 * Read only the exact bounded artifact promised by a completed graph job.
 *
 * Both streamed length and byte digest are checked. Content-Length alone is
 * insufficient: a chunked response may omit it, and a proxy may change it.
 *
 * @param response - Successful fixed-route HTTP response, still unread.
 * @param job - Checked completed actor-owned status retained before the request.
 * @returns Exact export bytes and their byte-verified, schema-checked scientific result.
 * @throws {Error} When headers, length, digest or result data disagree.
 */
export async function readGraphJobArtifact(response: Response, job: GraphJob): Promise<GraphJobArtifact> {
  const receipt = job.result;
  if (job.status !== "completed" || receipt === null || !Number.isSafeInteger(receipt.result_size_bytes)
    || receipt.result_size_bytes <= 0 || receipt.result_size_bytes > GRAPH_RESULT_MAX_BYTES) {
    await response.body?.cancel();
    throw new Error("Graph job has no bounded complete result.");
  }
  const expected = receipt.result_size_bytes;
  const length = response.headers.get("Content-Length");
  if (response.headers.get("Content-Type")?.split(";", 1)[0]?.trim().toLowerCase() !== "application/json"
    || response.headers.get("X-Studio-Artifact-SHA256") !== receipt.result_sha256
    || response.headers.get("X-Studio-Artifact-Size") !== String(expected)
    || length !== null && length !== String(expected) || response.body === null) {
    await response.body?.cancel();
    throw new Error("Graph result response does not match its declared artifact.");
  }
  const capturedJob = { ...job, admission: { ...job.admission }, result: { ...receipt } };
  const reader = response.body.getReader();
  const bytes = new Uint8Array(expected);
  let size = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (size + value.length > expected) throw new Error("Graph result response exceeds its declared size.");
      bytes.set(value, size);
      size += value.length;
    }
    if (size !== expected) throw new Error("Graph result response is incomplete.");
    return { bytes, result: await parseGraphJobResult(bytes, capturedJob) };
  } finally {
    await reader.cancel();
    reader.releaseLock();
  }
}
