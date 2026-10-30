// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Candidate lifecycle through real HTTP and process workers
// @vitest-environment happy-dom

import { readFileSync } from "node:fs";
import { once } from "node:events";
import { createServer, request as forwardRequest } from "node:http";
import { resolve } from "node:path";
import { Window } from "happy-dom";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { cancelCandidateJob, candidateReviewPacket, getCandidateJob, listCandidateJobs, simulateCandidate } from "../src/api/candidatesApi";
import { candidateJobActive } from "../src/candidateJobs";
import { startCandidateTestServer, type CandidateTestServer } from "../src/candidateTestServer";
import { useCandidateJobs } from "../src/useCandidateJobs";

const candidate: unknown = JSON.parse(readFileSync(resolve(import.meta.dirname, "./fixtures/SlowAdaptationAdEx.candidate.json"), "utf8"));
let server: CandidateTestServer;
let root: Root;
let container: HTMLDivElement;
let observation: ReturnType<typeof useCandidateJobs> | null = null;
const dom = window as unknown as Window;

/** Expose the real hook's current render to lifecycle assertions. */
function Observer() {
  observation = useCandidateJobs();
  return <p>{observation.selected?.status}</p>;
}

/** Require an actual mounted render. */
function current(): ReturnType<typeof useCandidateJobs> {
  if (observation === null) throw new Error("Candidate observer is not mounted.");
  return observation;
}

/** A real forwarding socket with one production response retained for a race. */
interface ResponseGate {
  origin: string;
  held: () => boolean;
  release: (disconnect?: boolean) => void;
  stop: () => Promise<void>;
}

/** Forward real HTTP unchanged, delaying or disconnecting one actual response. */
async function responseGate(method: string, path: string): Promise<ResponseGate> {
  let pending: { send: () => void; disconnect: () => void } | null = null;
  let intercepted = false;
  const proxy = createServer((incoming, outgoing) => {
    const target = new URL(incoming.url ?? "/", server.origin);
    const forwarded = forwardRequest(target, {
      method: incoming.method,
      headers: { ...incoming.headers, host: target.host, origin: server.origin, connection: "close" },
    }, (response) => {
      const chunks: Buffer[] = [];
      response.on("data", (chunk: Buffer) => { chunks.push(chunk); });
      response.once("end", () => {
        const send = () => {
          outgoing.writeHead(response.statusCode ?? 502, response.headers);
          outgoing.end(Buffer.concat(chunks));
        };
        if (!intercepted && incoming.method === method && target.pathname === path) {
          intercepted = true;
          pending = { send, disconnect: () => { outgoing.destroy(); } };
        } else send();
      });
    });
    forwarded.once("error", () => { outgoing.destroy(); });
    incoming.pipe(forwarded);
  });
  proxy.listen(0, "127.0.0.1");
  await once(proxy, "listening");
  const address = proxy.address();
  if (address === null || typeof address === "string") throw new Error("No response-gate socket.");
  return {
    origin: `http://127.0.0.1:${String(address.port)}`,
    held: () => pending !== null,
    release: (disconnect = false) => {
      if (pending === null) throw new Error("No actual response is held.");
      if (disconnect) pending.disconnect(); else pending.send();
      pending = null;
    },
    stop: async () => {
      pending?.disconnect(); pending = null;
      proxy.closeAllConnections();
      await new Promise<void>((accept) => { proxy.close(() => { accept(); }); });
    },
  };
}

/** Observe asynchronous HTTP/React transitions under a finite test deadline. */
async function until(condition: () => boolean): Promise<void> {
  const deadline = Date.now() + 45_000;
  while (!condition()) {
    if (Date.now() >= deadline) throw new Error(`Candidate observation expired: ${current().error ?? "no transport error"}`);
    await act(async () => { await new Promise((accept) => setTimeout(accept, 50)); });
  }
}

beforeAll(async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  server = await startCandidateTestServer();
}, 65_000);
beforeEach(async () => {
  dom.happyDOM.setURL(server.origin);
  dom.happyDOM.settings.fetch.requestHeaders = [{ url: /^http:\/\/127\.0\.0\.1:[0-9]+\/api(?:\/|$)/, headers: { "sec-fetch-site": "same-origin" } }];
  container = document.createElement("div"); document.body.append(container);
  root = createRoot(container);
  await act(async () => { root.render(<Observer />); await dom.happyDOM.waitUntilComplete(); });
});
afterEach(async () => {
  dom.happyDOM.setURL(server.origin);
  await act(async () => { root.unmount(); });
  await dom.happyDOM.waitUntilComplete();
  for (const job of await listCandidateJobs()) {
    if (candidateJobActive(job)) {
      await cancelCandidateJob(job.job_id);
      const deadline = Date.now() + 30_000;
      while (candidateJobActive(await getCandidateJob(job.job_id))) {
        if (Date.now() >= deadline) throw new Error("Candidate worker was not reaped.");
        await new Promise((accept) => setTimeout(accept, 50));
      }
    }
  }
  container.remove(); observation = null;
});
afterAll(async () => { await server.stop(); });

describe("real candidate lifecycle", () => {
  it("reports a failed history observation and recovers without submitting work", async () => {
    const before = await listCandidateJobs();
    dom.happyDOM.setURL("http://127.0.0.1:1");
    await act(async () => { current().refresh(); });
    await until(() => current().error?.startsWith("Could not load candidate history") === true);
    expect(current().selected).toBeNull();
    dom.happyDOM.setURL(server.origin);
    await act(async () => { current().refresh(); });
    await until(() => current().error === null);
    expect((await listCandidateJobs()).map((job) => job.job_id)).toEqual(before.map((job) => job.job_id));
  });

  it("observes completion, recovers stored reviews and refuses a missing selection", async () => {
    await act(async () => { await current().cancel(); });
    const receipt = await simulateCandidate(candidate, 800, 20);
    await act(async () => { current().accept(receipt); });
    await until(() => current().selected?.status === "completed");
    expect(current().selected?.result).toHaveProperty("steps", 20);
    await until(() => current().history.some((job) => job.job_id === receipt.job.job_id));
    await act(async () => { await current().cancel(); });
    const review = await candidateReviewPacket(candidate);
    await act(async () => { current().accept(review); });
    await until(() => current().selected?.status === "completed");
    expect(current().selected?.result).toHaveProperty("packet_sha256");
    await act(async () => { await current().select(receipt.job.job_id); });
    expect(current().selected?.admission.candidate_sha256).toBe(receipt.job.admission.candidate_sha256);
    expect(current().selected?.result).toHaveProperty("steps", 20);
    await act(async () => { await current().select("sj_0000000000000000"); });
    expect(current().selected).toBeNull();
    expect(current().error).toContain("Could not observe candidate job");
  }, 90_000);

  it("cancels through the actual supervisor and keeps the cancelled state", async () => {
    const receipt = await simulateCandidate(candidate, 800, 100_000);
    await act(async () => { current().accept(receipt); await current().select(receipt.job.job_id); });
    await act(async () => { await current().cancel(); });
    await until(() => current().selected?.status === "cancelled");
    expect(current().selected?.result).toBeNull();
  }, 60_000);

  it("reports a real transport failure and retries observation without resubmission", async () => {
    const receipt = await simulateCandidate(candidate, 800, 100_000);
    await act(async () => { current().accept(receipt); });
    dom.happyDOM.setURL("http://127.0.0.1:1");
    await until(() => current().error !== null);
    expect(current().selected?.job_id).toBe(receipt.job.job_id);
    await act(async () => { await current().cancel(); });
    expect(current().error).toContain("Could not request cancellation");
    dom.happyDOM.setURL(server.origin);
    await act(async () => { current().refresh(); });
    await act(async () => { await current().cancel(); });
    await until(() => current().selected?.status === "cancelled");
    const records = await listCandidateJobs();
    expect(records.filter((job) => job.job_id === receipt.job.job_id)).toHaveLength(1);
  }, 60_000);

  it.each([false, true])("keeps a newer selection after an old status response (disconnect=%s)", async (disconnect) => {
    const first = await simulateCandidate(candidate, 800, 20);
    await act(async () => { current().accept(first); });
    await until(() => current().selected?.status === "completed");
    const second = await simulateCandidate(candidate, 0, 20);
    expect(current().history.some((job) => job.job_id === second.job.job_id)).toBe(false);
    const gate = await responseGate("GET", first.status_route);
    let pending = Promise.resolve();
    try {
      dom.happyDOM.setURL(gate.origin);
      await act(async () => { pending = current().select(first.job.job_id); });
      await until(gate.held);
      await act(async () => { await current().select(second.job.job_id); });
      expect(current().selected?.job_id).toBe(second.job.job_id);
      await act(async () => { gate.release(disconnect); await pending; });
      expect(current().selected?.job_id).toBe(second.job.job_id);
      expect(current().error).toBeNull();
    } finally { dom.happyDOM.setURL(server.origin); await gate.stop(); }
  }, 60_000);

  it("ignores a disconnected history response after a newer refresh succeeds", async () => {
    const before = await listCandidateJobs();
    const gate = await responseGate("GET", "/api/candidates/jobs");
    try {
      dom.happyDOM.setURL(gate.origin);
      await act(async () => { current().refresh(); });
      await until(gate.held);
      await act(async () => { current().refresh(); });
      await act(async () => { gate.release(true); await dom.happyDOM.waitUntilComplete(); });
      expect(current().error).toBeNull();
      expect(current().history.map((job) => job.job_id)).toEqual(before.map((job) => job.job_id));
    } finally { dom.happyDOM.setURL(server.origin); await gate.stop(); }
  }, 60_000);

  it.each([false, true])("ignores a superseded cancellation response (disconnect=%s)", async (disconnect) => {
    const first = await simulateCandidate(candidate, 800, 100_000);
    await act(async () => { current().accept(first); });
    const gate = await responseGate("POST", first.status_route + "/cancel");
    let pending = Promise.resolve();
    try {
      dom.happyDOM.setURL(gate.origin);
      await act(async () => { pending = current().cancel(); });
      await until(gate.held);
      const second = await simulateCandidate(candidate, 0, 20);
      await act(async () => { await current().select(second.job.job_id); });
      await act(async () => { gate.release(disconnect); await pending; });
      expect(current().selected?.job_id).toBe(second.job.job_id);
      expect(current().error).toBeNull();
    } finally { dom.happyDOM.setURL(server.origin); await gate.stop(); }
  }, 60_000);

  it.each([false, true])("cannot revive a cancelled job from a late poll (disconnect=%s)", async (disconnect) => {
    const first = await simulateCandidate(candidate, 800, 100_000);
    const gate = await responseGate("GET", first.status_route);
    try {
      dom.happyDOM.setURL(gate.origin);
      await act(async () => { current().accept(first); });
      await until(gate.held);
      await act(async () => { await current().cancel(); });
      gate.release(disconnect);
      await until(() => current().selected?.status === "cancelled");
      expect(current().error).toBeNull();
      expect(current().selected?.job_id).toBe(first.job.job_id);
    } finally { dom.happyDOM.setURL(server.origin); await gate.stop(); }
  }, 60_000);
});
