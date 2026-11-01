// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Graph client through real HTTP and supervised workers
// @vitest-environment happy-dom

import { Window } from "happy-dom";
import { afterAll, afterEach, beforeAll, beforeEach, expect, it } from "vitest";
import { cancelGraphJob, downloadGraphJobArtifact, getGraphJob, listGraphJobs, prepareGraphJob, submitGraphJob } from "../src/api/graphJobsApi";
import { setStudioAuthToken } from "../src/api/http";
import type { NetworkGraph } from "../src/api/types";
import { graphJobActive, type GraphJob } from "../src/graphJobs";
import { startCandidateTestServer, type CandidateTestServer } from "../src/candidateTestServer";

let server: CandidateTestServer | null = null;
const dom = window as unknown as Window;

/** Require the actually started and exclusively owned backend. */
function backend(): CandidateTestServer {
  if (server === null) throw new Error("Graph backend is not running.");
  return server;
}

/** Build ordinary editable inputs; numerical execution stays in the real Python worker. */
function graph(count = 2, duration = 20, dt = 0.1): NetworkGraph {
  return {
    populations: [{ id: "e", type: "population", label: "Exc", model: "SCLapicqueLIFNeuron",
      count, neuron_type: "excitatory", params: { tau: 10 }, position: { x: 100, y: 100 },
      drive: { kind: "constant", current: 2 } }],
    projections: [], duration, dt, seed: 42,
  };
}

/** Wait for actual supervisor state rather than manufacturing a completed response. */
async function settled(jobId: string): Promise<GraphJob> {
  const deadline = Date.now() + 60_000;
  for (;;) {
    const job = await getGraphJob(jobId, "local");
    if (!graphJobActive(job)) return job;
    if (Date.now() >= deadline) throw new Error("Graph worker did not settle.");
    await new Promise((accept) => setTimeout(accept, 50));
  }
}

beforeAll(async () => { server = await startCandidateTestServer(); }, 65_000);
beforeEach(() => {
  dom.happyDOM.setURL(backend().origin);
  dom.happyDOM.settings.fetch.requestHeaders = [{ url: /^http:\/\/127\.0\.0\.1:[0-9]+\/api(?:\/|$)/,
    headers: { "sec-fetch-site": "same-origin" } }];
  setStudioAuthToken(null);
});
afterEach(async () => {
  setStudioAuthToken(null);
  dom.happyDOM.setURL(backend().origin);
  for (const job of await listGraphJobs("local")) {
    if (graphJobActive(job)) {
      await cancelGraphJob(job.job_id, "local");
      expect(graphJobActive(await settled(job.job_id))).toBe(false);
    }
  }
  await dom.happyDOM.waitUntilComplete();
});
afterAll(async () => { if (server !== null) await server.stop(); });

it("prepares without allocation, submits exact captured bytes and recovers the same durable job", async () => {
  const before = await listGraphJobs("local");
  const request = graph();
  const prepared = await prepareGraphJob(request);
  expect(await listGraphJobs("local")).toHaveLength(before.length);
  request.duration = 999;
  const receipt = await submitGraphJob(prepared, "local", "graph-client-exact-retry");
  const completed = await settled(receipt.job.job_id);
  expect(completed.status).toBe("completed");
  expect(completed.result).toMatchObject({ n_steps: 200, n_total: 2 });
  const { result, bytes } = await downloadGraphJobArtifact(completed);
  expect(bytes.length).toBe(completed.result?.result_size_bytes);
  expect(result.n_total).toBe(2);
  expect(result.spec.graph_sha256).toBe(completed.admission.graph_sha256);
  expect(result.populations[0]?.events.step).toHaveLength(result.n_spikes);
  await expect(downloadGraphJobArtifact({ ...completed, status: "running" })).rejects.toThrow("no complete result");
  await expect(downloadGraphJobArtifact({ ...completed, result: null })).rejects.toThrow("no complete result");
  await expect(downloadGraphJobArtifact({ ...completed, job_id: "sj_0000000000000000" })).rejects.toThrow("graph_job_not_found");
  const retried = await submitGraphJob(prepared, "local", "graph-client-exact-retry");
  expect(retried.job.job_id).toBe(completed.job_id);
  expect((await cancelGraphJob(completed.job_id, "local")).status).toBe("completed");
  const changed = await prepareGraphJob(graph(2, 21));
  await expect(submitGraphJob(changed, "local", "graph-client-exact-retry")).rejects.toThrow("graph_job_idempotency_conflict");
  expect(await listGraphJobs("local")).toHaveLength(before.length + 1);
  await expect(listGraphJobs("another-actor")).rejects.toThrow("Invalid graph job record or actor");
}, 90_000);

it("stops malformed, changed-input, aborted and switched-account submissions before allocation", async () => {
  const before = await listGraphJobs("local");
  const prepared = await prepareGraphJob(graph());
  await expect(submitGraphJob(prepared, "local", "key with spaces")).rejects.toThrow("retry key");
  const changed = { ...prepared, request: { ...prepared.request, json: prepared.request.json.replace('"duration":20', '"duration":21') } };
  await expect(submitGraphJob(changed, "local", "changed-input")).rejects.toThrow("no longer matches");
  const abort = new AbortController();
  abort.abort();
  await expect(submitGraphJob(prepared, "local", "aborted", abort.signal)).rejects.toThrow();
  const switching = submitGraphJob(prepared, "local", "switching-account");
  setStudioAuthToken("different-session");
  await expect(switching).rejects.toThrow("account changed");
  setStudioAuthToken(null);
  await expect(prepareGraphJob(graph(1, 2_000_001, 1))).rejects.toThrow();
  await expect(prepareGraphJob(graph(), abort.signal)).rejects.toThrow();
  await expect(getGraphJob("../outside", "local")).rejects.toThrow("Invalid graph job identifier");
  await expect(getGraphJob("sj_0000000000000000", "local")).rejects.toThrow("graph_job_not_found");
  expect(await listGraphJobs("local")).toHaveLength(before.length);
}, 90_000);

it("cancels an actually running above-sync graph through the authority", async () => {
  const prepared = await prepareGraphJob(graph(2000, 1000, 1));
  const receipt = await submitGraphJob(prepared, "local", "running-graph-cancel");
  const deadline = Date.now() + 30_000;
  while ((await getGraphJob(receipt.job.job_id, "local")).status === "pending") {
    if (Date.now() >= deadline) throw new Error("Graph worker never started.");
    await new Promise((accept) => setTimeout(accept, 20));
  }
  expect((await getGraphJob(receipt.job.job_id, "local")).status).toBe("running");
  const cancelling = await cancelGraphJob(receipt.job.job_id, "local");
  expect(["cancelling", "cancelled"]).toContain(cancelling.status);
  const final = await settled(receipt.job.job_id);
  expect(final.status).toBe("cancelled");
  expect(final.result).toBeNull();
  await expect(downloadGraphJobArtifact(final)).rejects.toThrow("no complete result");
}, 90_000);
