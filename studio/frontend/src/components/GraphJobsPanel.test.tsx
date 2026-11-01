// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Visible graph job controls through the real service
// @vitest-environment happy-dom

import { createHash } from "node:crypto";
import { Window } from "happy-dom";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterAll, afterEach, beforeAll, beforeEach, expect, it, vi } from "vitest";
import { setStudioAuthToken } from "../api/http";
import { cancelGraphJob, getGraphJob, listGraphJobs } from "../api/graphJobsApi";
import { graphJobActive } from "../graphJobs";
import { graphWorkspaceId } from "../graphJobRecovery";
import { startCandidateTestServer, type CandidateTestServer } from "../candidateTestServer";
import { useStudioStore } from "../stores/studio";
import { studioInitialData } from "../stores/studioInitialState";
import GraphJobsPanel from "./GraphJobsPanel";

let server: CandidateTestServer;
let root: Root;
let container: HTMLDivElement;
const dom = window as unknown as Window;

/** Require a visible production control by its accessible text. */
function button(label: string): HTMLButtonElement {
  const found = Array.from(container.querySelectorAll("button")).find((element) => element.textContent === label);
  if (found === undefined) throw new Error(`Graph control is missing: ${label}`);
  return found;
}

/** Await actual UI/network transitions under React's ordinary update boundary. */
async function until(condition: () => boolean): Promise<void> {
  const deadline = Date.now() + 45_000;
  while (!condition()) {
    if (Date.now() >= deadline) throw new Error(`Graph panel expired: ${container.textContent}`);
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 40)); });
  }
}

/** Make the real development session available, as the normal startup session request does. */
async function ready(): Promise<void> {
  await act(async () => { useStudioStore.setState({ authSession: { authenticated: false, principal_id: null, roles: [] } }); });
  await until(() => !button("Run background job").disabled);
}

beforeAll(async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  server = await startCandidateTestServer();
}, 65_000);
beforeEach(async () => {
  dom.happyDOM.setURL(server.origin);
  dom.happyDOM.settings.fetch.requestHeaders = [{ url: /^http:\/\/127\.0\.0\.1:[0-9]+\/api(?:\/|$)/,
    headers: { "sec-fetch-site": "same-origin" } }];
  localStorage.clear(); sessionStorage.clear(); setStudioAuthToken(null);
  useStudioStore.setState({ ...studioInitialData, graphWorkspaceId: graphWorkspaceId(), graphContextToken: { restored: false },
    dt: 0.1, duration: 20, seed: 42,
    graphPopulations: [{ id: "e", type: "population", label: "Exc", model: "SCLapicqueLIFNeuron",
      count: 2, neuron_type: "excitatory", params: { tau: 10 }, position: { x: 10, y: 20 }, drive: { kind: "constant", current: 2 } }],
    graphProjections: [] });
  container = document.createElement("div"); document.body.append(container); root = createRoot(container);
  await act(async () => { root.render(<GraphJobsPanel />); });
});
afterEach(async () => {
  await act(async () => { root.unmount(); });
  vi.restoreAllMocks(); dom.happyDOM.setURL(server.origin); setStudioAuthToken(null);
  await dom.happyDOM.waitUntilComplete();
  for (const job of await listGraphJobs("local")) {
    if (graphJobActive(job)) {
      await cancelGraphJob(job.job_id, "local");
      const deadline = Date.now() + 30_000;
      while (graphJobActive(await getGraphJob(job.job_id, "local"))) {
        if (Date.now() >= deadline) throw new Error("Panel worker did not reap.");
        await new Promise((resolve) => setTimeout(resolve, 40));
      }
    }
  }
  container.remove();
});
afterAll(async () => { await server.stop(); });

it("gates the account, runs actual work and exports the exact verified original bytes", async () => {
  expect(button("Run background job").disabled).toBe(true);
  expect(container.textContent).toContain("Checking the Studio account");
  await ready();
  await act(async () => { button("Run background job").click(); });
  await until(() => !button("Export verified result").disabled && useStudioStore.getState().graphSimResult !== null);
  expect(container.querySelector('[data-testid="selected-graph-job"]')?.textContent).toContain("completed");
  expect(useStudioStore.getState().graphSimResult?.n_total).toBe(2);
  const observedUrls = vi.spyOn(URL, "createObjectURL");
  const observedClicks = vi.spyOn(HTMLAnchorElement.prototype, "click");
  await act(async () => { button("Export verified result").click(); dom.happyDOM.setURL(server.origin); });
  const blob = observedUrls.mock.calls.at(-1)?.[0];
  if (!(blob instanceof Blob)) throw new Error("Real download did not receive a Blob");
  const id = useStudioStore.getState().graphJobRecovery?.selectedJobId;
  if (id === null || id === undefined) throw new Error("No retained selected job");
  const job = await getGraphJob(id, "local");
  expect(createHash("sha256").update(Buffer.from(await blob.arrayBuffer())).digest("hex")).toBe(job.result?.result_sha256);
  const anchor: unknown = observedClicks.mock.contexts.at(-1);
  if (!(anchor instanceof HTMLAnchorElement)) throw new Error("No actual download anchor");
  expect(anchor.download).toBe("network-result.json");
  const adopted = useStudioStore.getState().graphSimResult;
  observedUrls.mockImplementationOnce(() => { throw new DOMException("Browser object URL was refused.", "NotSupportedError"); });
  await act(async () => { button("Export verified result").click(); });
  expect(container.querySelector('[role="alert"]')?.textContent).toContain("Export failed");
  expect(useStudioStore.getState().graphSimResult).toBe(adopted);
  expect(button("Export verified result").disabled).toBe(false);
  await act(async () => { button("Export verified result").click(); dom.happyDOM.setURL(server.origin); });
  expect(container.querySelector('[role="alert"]')).toBeNull();
  const retried = observedUrls.mock.calls.at(-1)?.[0];
  if (!(retried instanceof Blob)) throw new Error("Retry did not receive a real download Blob");
  expect(createHash("sha256").update(Buffer.from(await retried.arrayBuffer())).digest("hex")).toBe(job.result?.result_sha256);
  await act(async () => { useStudioStore.getState().updatePopulation("e", { params: { tau: 11 } }); });
  expect(container.textContent).toContain("previous job result was withdrawn");
  await act(async () => { button("Refresh jobs").click(); });
  await until(() => container.textContent.includes("different graph"));
  expect(button("Export verified result").disabled).toBe(false);
  await act(async () => { button("Forget local selection").click(); });
  expect(container.querySelector('[data-testid="selected-graph-job"]')).toBeNull();
  await until(() => container.querySelector('ul[aria-label="Stored graph calculations"] button') !== null);
  await act(async () => { container.querySelector<HTMLButtonElement>('ul[aria-label="Stored graph calculations"] button')?.click(); });
  await until(() => !button("Export verified result").disabled);
}, 90_000);

it("reports corrupt browser recovery and resets only its local pointer before allowing another job", async () => {
  const workspace = useStudioStore.getState().graphWorkspaceId;
  localStorage.setItem(`sc-neurocore-graph-job:${workspace}:local`, "{corrupt");
  await act(async () => { useStudioStore.setState({ authSession: { authenticated: false, principal_id: null, roles: [] } }); });
  await until(() => container.querySelector('[role="alert"]') !== null);
  expect(button("Run background job").disabled).toBe(true);
  expect(container.textContent).toContain("Recovery could not be read");
  await act(async () => { button("Forget local selection").click(); });
  expect(button("Run background job").disabled).toBe(false);
  expect(container.querySelector('[role="alert"]')).toBeNull();
  await act(async () => { useStudioStore.setState({ graphPopulations: [] }); });
  expect(button("Run background job").disabled).toBe(true);
});

it("cancels actual running background work through the visible authority control", async () => {
  await act(async () => { useStudioStore.setState({ dt: 1, duration: 1000 }); useStudioStore.getState().updatePopulation("e", { count: 2000 }); });
  await ready();
  await act(async () => { button("Run background job").click(); });
  await until(() => container.querySelector('[data-testid="selected-graph-job"]')?.textContent.includes(": running") === true);
  await act(async () => { button("Cancel selected job").click(); });
  await until(() => container.querySelector('[data-testid="selected-graph-job"]')?.textContent.includes(": cancelled") === true);
  expect(button("Export verified result").disabled).toBe(true);
  expect(useStudioStore.getState().graphSimResult).toBeNull();
}, 90_000);

it("shows an actual oversized complete-result failure without exposing a partial export", async () => {
  await act(async () => {
    useStudioStore.setState({ dt: 1, duration: 1000 });
    useStudioStore.getState().updatePopulation("e", { count: 1000, drive: { kind: "constant", current: 1e6 } });
  });
  await ready();
  await act(async () => { button("Run background job").click(); });
  await until(() => container.querySelector('[data-testid="selected-graph-job"]')?.textContent.includes(": failed") === true);
  expect(container.textContent).toContain("Graph job JSON exceeds its byte limit.");
  expect(button("Export verified result").disabled).toBe(true);
  expect(useStudioStore.getState().graphSimResult).toBeNull();
}, 90_000);
