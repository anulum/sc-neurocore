// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Actual graph integration fixture and bounded response relay
// @vitest-environment happy-dom

import { once } from "node:events";
import { pbkdf2 } from "node:crypto";
import { createServer, get as getHttp, request as forward } from "node:http";
import { Window } from "happy-dom";
import { act, createElement, StrictMode, useLayoutEffect } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterAll, afterEach, beforeAll, beforeEach, vi } from "vitest";
import { cancelGraphJob, getGraphJob, listGraphJobs } from "../src/api/graphJobsApi";
import { setStudioAuthToken } from "../src/api/http";
import { graphJobActive } from "../src/graphJobs";
import { graphWorkspaceId } from "../src/graphJobRecovery";
import { startCandidateTestServer, type CandidateTestServer } from "../src/candidateTestServer";
import { useStudioStore } from "../src/stores/studio";
import { studioInitialData } from "../src/stores/studioInitialState";
import { useGraphJobs } from "../src/useGraphJobs";

export let server: CandidateTestServer;
export let root: Root;
export let container: HTMLDivElement;
let observation: ReturnType<typeof useGraphJobs> | null = null;
export const dom = window as unknown as Window;
const relays: ResponseGate[] = [];
export const committedViews: { owner: string | null; workspace: string; history: number; selected: string | null; artifact: boolean }[] = [];

/** Expose the actual mounted lifecycle while keeping every HTTP request real. */
function Observer() {
  observation = useGraphJobs();
  const { history, selected, artifact } = observation;
  const { authSession, graphWorkspaceId: workspace } = useStudioStore.getState();
  useLayoutEffect(() => {
    committedViews.push({ owner: authSession?.principal_id ?? null, workspace,
      history: history.length, selected: selected?.job_id ?? null, artifact: artifact !== null });
  });
  return createElement("p", null, observation.selected?.status);
}

/** Require one actual mounted render. */
export function current(): ReturnType<typeof useGraphJobs> {
  if (observation === null) throw new Error("Graph observer is not mounted.");
  return observation;
}

/** One real response retained after the backend already acted. */
interface ResponseGate {
  origin: string;
  held: () => boolean;
  release: (disconnect?: boolean) => void;
  stop: () => Promise<void>;
}

/** Delay or disconnect one actual backend response without replacing its contents. */
export async function responseGate(method: string, matches: (path: string) => boolean): Promise<ResponseGate> {
  let pending: { send: () => void; disconnect: () => void } | null = null;
  let intercepted = false;
  const proxy = createServer((incoming, outgoing) => {
    const target = new URL(incoming.url ?? "/", server.origin);
    const request = forward(target, { method: incoming.method,
      headers: { ...incoming.headers, host: target.host, origin: server.origin, connection: "close" } }, (response) => {
      const chunks: Buffer[] = [];
      response.on("data", (chunk: Buffer) => { chunks.push(chunk); });
      response.once("end", () => {
        const send = () => { outgoing.writeHead(response.statusCode ?? 502, response.headers); outgoing.end(Buffer.concat(chunks)); };
        if (!intercepted && incoming.method === method && matches(target.pathname)) {
          intercepted = true; pending = { send, disconnect: () => { outgoing.destroy(); } };
        } else send();
      });
    });
    request.once("error", () => { outgoing.destroy(); });
    incoming.pipe(request);
  });
  proxy.listen(0, "127.0.0.1"); await once(proxy, "listening");
  const address = proxy.address();
  if (address === null || typeof address === "string") throw new Error("No graph response-gate socket.");
  const gate: ResponseGate = {
    origin: `http://127.0.0.1:${String(address.port)}`, held: () => pending !== null,
    release: (disconnect = false) => {
      if (pending === null) throw new Error("No graph response is held.");
      if (disconnect) pending.disconnect(); else pending.send();
      pending = null;
    },
    stop: async () => {
      pending?.disconnect(); pending = null; proxy.closeAllConnections();
      await new Promise<void>((resolve) => { proxy.close(() => { resolve(); }); });
    },
  };
  relays.push(gate); return gate;
}

/** Wait for actual public state under one finite deadline and React's normal update boundary. */
export async function until(condition: () => boolean): Promise<void> {
  const deadline = Date.now() + 45_000;
  while (!condition()) {
    if (Date.now() >= deadline) throw new Error(`Graph transition expired: ${current().error ?? current().resultError ?? "no reported error"}`);
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 40)); });
  }
}

/** Read the authority directly while the browser may be observing a forwarding origin. */
export async function authorityJobCount(): Promise<number> {
  let count = 0;
  await act(async () => {
    const body: unknown = await new Promise((resolve, reject) => {
      const request = getHttp(server.origin + "/api/graph/jobs?limit=100", { headers: { "sec-fetch-site": "same-origin" } }, (response) => {
        const chunks: Buffer[] = [];
        response.on("data", (chunk: Buffer) => { chunks.push(chunk); });
        response.once("end", () => {
          try {
            if (response.statusCode !== 200) throw new Error("Authority history was refused");
            const value: unknown = JSON.parse(Buffer.concat(chunks).toString("utf8")); resolve(value);
          } catch (failure) { reject(failure instanceof Error ? failure : new Error(String(failure))); }
        });
      });
      request.setTimeout(5000, () => { request.destroy(new Error("Authority history timed out")); });
      request.once("error", reject);
    });
    if (body === null || typeof body !== "object" || !("jobs" in body) || !Array.isArray(body.jobs)) throw new Error("No authority history");
    count = body.jobs.length;
  });
  return count;
}

beforeAll(async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  server = await startCandidateTestServer();
}, 65_000);
beforeEach(async () => {
  dom.happyDOM.setURL(server.origin);
  dom.happyDOM.settings.fetch.requestHeaders = [{ url: /^http:\/\/127\.0\.0\.1:[0-9]+\/api(?:\/|$)/,
    headers: { "sec-fetch-site": "same-origin" } }];
  localStorage.clear(); sessionStorage.clear(); setStudioAuthToken(null); committedViews.length = 0;
  useStudioStore.setState({ ...studioInitialData, authSession: { authenticated: false, principal_id: null, roles: [] },
    graphWorkspaceId: graphWorkspaceId(), graphContextToken: { restored: false },
    sourceMode: "model", selectedModelName: "PoissonNeuron", dt: 0.1, duration: 20, seed: 42,
    graphPopulations: [{ id: "e", type: "population", label: "Exc", model: "SCLapicqueLIFNeuron",
      count: 2, neuron_type: "excitatory", params: { tau: 10 }, position: { x: 10, y: 20 }, drive: { kind: "constant", current: 2 } }],
    graphProjections: [] });
  container = document.createElement("div"); document.body.append(container); root = createRoot(container);
  await act(async () => { root.render(createElement(StrictMode, null, createElement(Observer))); await dom.happyDOM.waitUntilComplete(); });
});
afterEach(async () => {
  await act(async () => { root.unmount(); });
  for (const relay of relays.splice(0)) await relay.stop();
  dom.happyDOM.setURL(server.origin); setStudioAuthToken(null);
  await dom.happyDOM.waitUntilComplete();
  for (const job of await listGraphJobs("local")) {
    if (graphJobActive(job)) {
      await cancelGraphJob(job.job_id, "local");
      const deadline = Date.now() + 30_000;
      while (graphJobActive(await getGraphJob(job.job_id, "local"))) {
        if (Date.now() >= deadline) throw new Error("Graph worker did not reap.");
        await new Promise((resolve) => setTimeout(resolve, 40));
      }
    }
  }
  container.remove(); observation = null;
});
afterAll(async () => { await server.stop(); });


/** Hold actual WebCrypto behind bounded native CPU work, observing its real promise without replacing it. */
export async function whileDigestPending(release: () => void, changeContext: () => void): Promise<void> {
  const threads = Number(process.env.UV_THREADPOOL_SIZE ?? "4");
  if (!Number.isInteger(threads) || threads < 1 || threads > 8) throw new Error("Graph crypto race needs a bounded native thread pool.");
  const busy = Promise.all(Array.from({ length: threads }, () => new Promise<void>((resolve, reject) => {
    pbkdf2("graph-race", "native-queue", 1_000_000, 32, "sha256", (failure) => {
      if (failure !== null) reject(failure); else resolve();
    });
  })));
  const hashes = vi.spyOn(crypto.subtle, "digest");
  try {
    await act(async () => { release(); });
    await until(() => hashes.mock.results.some((result) => result.type === "return"));
    const result = hashes.mock.results.find((result) => result.type === "return");
    if (result?.type !== "return") throw new Error("No actual WebCrypto digest was observed.");
    const completion = { settled: false };
    void result.value.then(() => { completion.settled = true; }, () => { completion.settled = true; });
    await act(async () => { await new Promise<void>((resolve) => setImmediate(resolve)); });
    if (completion.settled) throw new Error("Native digest completed before the graph context race could be exercised.");
    await act(async () => { changeContext(); });
  } finally {
    try {
      await act(async () => {
        await busy;
        let observed = -1;
        while (observed !== hashes.mock.results.length) {
          observed = hashes.mock.results.length;
          await Promise.all(hashes.mock.results.filter((result) => result.type === "return").map((result) => result.value));
          await new Promise<void>((resolve) => setImmediate(resolve));
        }
      });
    } finally { hashes.mockRestore(); }
  }
}
