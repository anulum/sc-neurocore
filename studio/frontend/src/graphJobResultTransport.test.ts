// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Graph artifact transport through actual HTTP streams
// @vitest-environment node

import { readFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, expect, it } from "vitest";
import { GRAPH_RESULT_MAX_BYTES } from "./graphJobResult";
import { readGraphJobArtifact } from "./graphJobResultTransport";
import { parseGraphJob, type GraphJob } from "./graphJobs";

let server: Server;
let origin: string;
let artifact: string;
let job: GraphJob;

beforeAll(async () => {
  const sample: unknown = JSON.parse(readFileSync(new URL("./test-data/graphJobContract.json", import.meta.url), "utf8"));
  if (sample === null || typeof sample !== "object" || !("artifact_text" in sample) || typeof sample.artifact_text !== "string"
    || !("completed" in sample)) throw new Error("Missing native corpus");
  artifact = sample.artifact_text;
  job = await parseGraphJob(sample.completed, "alice");
  if (job.result === null) throw new Error("Missing complete receipt");
  const receipt = { ...job.result };
  server = createServer((request, response) => {
    const fault = request.url;
    response.setHeader("Content-Type", fault === "/type" ? "text/html" : "application/json; charset=utf-8");
    response.setHeader("X-Studio-Artifact-SHA256", fault === "/sha" ? "0".repeat(64) : receipt.result_sha256);
    response.setHeader("X-Studio-Artifact-Size", fault === "/size" ? "1" : String(receipt.result_size_bytes));
    if (fault === "/missing-type") response.removeHeader("Content-Type");
    if (fault === "/length") response.setHeader("Content-Length", String(receipt.result_size_bytes + 1));
    if (fault === "/no-body") { response.writeHead(204); response.end(); return; }
    const bytes = Buffer.from(artifact);
    response.write(bytes.subarray(0, 100));
    if (fault === "/short") response.end();
    else if (fault === "/broken") {
      setImmediate(() => response.destroy(new Error("Owned graph response interrupted")));
    } else {
      response.write(bytes.subarray(100));
      if (fault === "/long") response.write(" ");
      response.end();
    }
  });
  await new Promise<void>((resolve, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", resolve); });
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("Missing exclusive HTTP port");
  origin = `http://127.0.0.1:${String(address.port)}`;
});

afterAll(async () => {
  server.closeAllConnections();
  await new Promise<void>((resolve, reject) => { server.close((error) => { if (error === undefined) resolve(); else reject(error); }); });
});

it("reads the entire chunked native artifact with no Content-Length assumption", async () => {
  const response = await fetch(origin);
  expect(response.headers.get("Content-Length")).toBeNull();
  const { result, bytes } = await readGraphJobArtifact(response, job);
  expect(new TextDecoder().decode(bytes)).toBe(artifact);
  expect(result.n_spikes).toBe(4);
  expect(result.populations).toHaveLength(3);
});

it.each(["type", "missing-type", "sha", "size", "length", "no-body"])("refuses incorrect %s before accepting a body", async (fault) => {
  await expect(readGraphJobArtifact(await fetch(`${origin}/${fault}`), job)).rejects.toThrow("declared artifact");
});

it.each(["short", "long", "broken"])("refuses an actual %s transfer", async (fault) => {
  await expect(readGraphJobArtifact(await fetch(`${origin}/${fault}`), job)).rejects.toThrow();
});

it("does not allocate a result buffer for a non-completed, absent or oversized receipt", async () => {
  if (job.result === null) throw new Error("Missing result");
  const receipt = job.result;
  for (const invalid of [{ ...job, status: "running" as const }, { ...job, result: null },
    ...[0, 0.5, GRAPH_RESULT_MAX_BYTES + 1].map((size) => ({ ...job, result: { ...receipt, result_size_bytes: size } }))]) {
    await expect(readGraphJobArtifact(await fetch(origin), invalid)).rejects.toThrow("no bounded complete result");
  }
  await expect(readGraphJobArtifact(await fetch(`${origin}/no-body`), { ...job, result: null })).rejects.toThrow("no bounded complete result");
});
