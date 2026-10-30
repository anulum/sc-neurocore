// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Catalogue proposal adoption and real HTTP response races
// @vitest-environment happy-dom

import { once } from "node:events";
import { createServer, request as forwardRequest } from "node:http";
import { Window } from "happy-dom";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterAll, afterEach, beforeAll, beforeEach, expect, it } from "vitest";
import { startCandidateTestServer, type CandidateTestServer } from "../candidateTestServer";
import CandidateCatalogueStart from "./CandidateCatalogueStart";

let server: CandidateTestServer;
let container: HTMLDivElement;
let root: Root;
let adopted: string[];
const dom = window as unknown as Window;
beforeAll(async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  server = await startCandidateTestServer();
  dom.happyDOM.settings.fetch.requestHeaders = [{ url: /^http:\/\/127\.0\.0\.1:[0-9]+\/api(?:\/|$)/,
    headers: { "sec-fetch-site": "same-origin" } }];
}, 40_000);
beforeEach(() => {
  dom.happyDOM.setURL(server.origin);
  adopted = [];
  container = document.createElement("div"); document.body.append(container);
  root = createRoot(container);
});
afterEach(async () => {
  await act(async () => { root.unmount(); });
  container.remove(); await dom.happyDOM.waitUntilComplete();
});
afterAll(async () => { await server.stop(); });

/** Render actual selected-identity and current-draft component inputs. */
async function render(modelName = "AdExNeuron", draft = ""): Promise<void> {
  await act(async () => { root.render(<CandidateCatalogueStart modelName={modelName} draft={draft}
    onAdopt={(text) => { adopted.push(text); }} />); });
}

/** Locate the actual visible action by its exact label. */
function button(label: string): HTMLButtonElement {
  const result = [...container.querySelectorAll("button")].find((p) => p.textContent === label);
  if (result === undefined) throw new Error(`Missing actual button ${label}`);
  return result;
}

/** Dispatch a real DOM click without waiting for a deliberately held HTTP response. */
async function click(label: string): Promise<void> {
  await act(async () => { button(label).click(); });
}

/** Observe real HTTP/React changes within the unchanged startup budget. */
async function until(condition: () => boolean): Promise<void> {
  const end = Date.now() + 30_000;
  while (!condition()) {
    if (Date.now() > end) throw new Error("Catalogue candidate observation expired.");
    await act(async () => { await new Promise((done) => setTimeout(done, 25)); });
  }
}

/** Read the visible read-only proposal, without parsing and rewriting it. */
function preview(): string | undefined {
  return container.querySelector<HTMLTextAreaElement>("textarea")?.value;
}

/** Hold one actual upstream response for a selected-model or unmount race. */
async function responseGate(): Promise<{
  origin: string; held: () => boolean; release: (disconnect: boolean) => void; stop: () => Promise<void>;
}> {
  let pending: { send: () => void; disconnect: () => void } | null = null;
  const proxy = createServer((incoming, outgoing) => {
    const target = new URL(incoming.url ?? "/", server.origin);
    const forwarded = forwardRequest(target, { method: incoming.method,
      headers: { ...incoming.headers, host: target.host, origin: server.origin, connection: "close" } }, (response) => {
      const chunks: Buffer[] = [];
      response.on("data", (chunk: Buffer) => { chunks.push(chunk); });
      response.once("end", () => {
        const send = () => { outgoing.writeHead(response.statusCode ?? 502, response.headers); outgoing.end(Buffer.concat(chunks)); };
        if (target.pathname.endsWith("/AdExNeuron")) pending = { send, disconnect: () => { outgoing.destroy(); } };
        else send();
      });
    });
    forwarded.once("error", () => { outgoing.destroy(); }); incoming.pipe(forwarded);
  });
  proxy.listen(0, "127.0.0.1"); await once(proxy, "listening");
  const address = proxy.address();
  if (address === null || typeof address === "string") throw new Error("Missing owned relay socket.");
  return { origin: `http://127.0.0.1:${String(address.port)}`, held: () => pending !== null,
    release: (disconnect) => {
      if (pending === null) throw new Error("No actual response is held.");
      if (disconnect) pending.disconnect(); else pending.send(); pending = null;
    },
    stop: async () => {
      pending?.disconnect(); pending = null; proxy.closeAllConnections();
      await new Promise<void>((done) => { proxy.close(() => { done(); }); });
    } };
}

it("requires a selected model and never changes the workspace by preparing a preview", async () => {
  await render(""); expect(button("Prepare catalogue draft").disabled).toBe(true);
  expect(container.textContent).toContain("Selected catalogue model: none");
  await render(); await click("Prepare catalogue draft");
  expect(button("Preparing catalogue draft…").disabled).toBe(true);
  await until(() => preview() !== undefined);
  expect(adopted).toEqual([]);
  const expected = preview(); expect(expected).toContain('"parent": "AdExNeuron"');
  expect(container.textContent).toContain("any blank units before validation");
  await click("Use catalogue draft"); expect(adopted).toEqual([expected]); expect(preview()).toBeUndefined();
});

it("requires explicit replacement even when the current draft changes during a request", async () => {
  const gate = await responseGate();
  try {
    dom.happyDOM.setURL(gate.origin); await render("AdExNeuron", "first draft");
    await click("Prepare catalogue draft"); await until(gate.held);
    await render("AdExNeuron", "new unsaved draft"); expect(adopted).toEqual([]);
    gate.release(false); await until(() => preview() !== undefined);
    const expected = preview(); expect(adopted).toEqual([]);
    await click("Replace current candidate draft"); expect(adopted).toEqual([expected]);
  } finally { await gate.stop(); }
});

it("reports a real unsupported-model refusal and lets the author retry", async () => {
  await render("NoSuchCatalogueModel"); await click("Prepare catalogue draft");
  await until(() => container.querySelector('[role="alert"]') !== null);
  expect(container.textContent).toContain("Select a registered catalogue model.");
  expect(adopted).toEqual([]); await render(); await click("Prepare catalogue draft");
  await until(() => preview() !== undefined); expect(container.querySelector('[role="alert"]')).toBeNull();
});

it.each([false, true])("discards a late response after selection changes, disconnect=%s", async (disconnect) => {
  const gate = await responseGate();
  try {
    dom.happyDOM.setURL(gate.origin); await render(); await click("Prepare catalogue draft"); await until(gate.held);
    await render("PoissonNeuron"); await click("Prepare catalogue draft"); await until(() => preview() !== undefined);
    const current = preview(); gate.release(disconnect);
    await act(async () => { await dom.happyDOM.waitUntilComplete(); });
    expect(preview()).toBe(current); expect(current).toContain('"parent": "PoissonNeuron"');
    expect(container.querySelector('[role="alert"]')).toBeNull(); expect(adopted).toEqual([]);
    await render("AdExNeuron"); expect(preview()).toBeUndefined();
  } finally { await gate.stop(); }
});

it.each([false, true])("discards a completed request after unmount, disconnect=%s", async (disconnect) => {
  const gate = await responseGate();
  try {
    dom.happyDOM.setURL(gate.origin); await render(); await click("Prepare catalogue draft"); await until(gate.held);
    await act(async () => { root.render(<p>Candidate view closed</p>); }); gate.release(disconnect);
    await act(async () => { await dom.happyDOM.waitUntilComplete(); });
    expect(container.textContent).toBe("Candidate view closed"); expect(adopted).toEqual([]);
  } finally { await gate.stop(); }
});
