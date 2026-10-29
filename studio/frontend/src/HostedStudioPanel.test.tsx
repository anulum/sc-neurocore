// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — hosted entry and parent-message lifecycle regressions
// @vitest-environment happy-dom
// The parent window is a real same-document about:blank frame; happy-dom 21 loads frames only on request.
// @vitest-environment-options {"settings":{"enableIframePageLoading":true}}

import { act, StrictMode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import SnnStudioPanel from "./SnnStudioPanel";
import { requestHostFederation } from "./hostBridge";

/** A request captured at the actual parent-window communication boundary. */
interface Request { id: string; capability: string }
let root: Root | undefined;
let parent: Window;
let host: HTMLDivElement;
let requests: Request[];
let baselineTimers: number;

beforeEach(async () => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.useFakeTimers();
  const frame = document.createElement("iframe");
  document.body.append(frame);
  const frameWindow = frame.contentWindow;
  if (!frameWindow) throw new Error("Test parent missing");
  parent = frameWindow;
  vi.stubGlobal("parent", parent);
  requests = [];
  vi.spyOn(parent, "postMessage").mockImplementation((data: unknown) => {
    if (typeof data !== "object" || data === null || !("id" in data)
      || !("capability" in data) || typeof data.id !== "string"
      || typeof data.capability !== "string") throw new Error("Invalid bridge request");
    requests.push({ id: data.id, capability: data.capability });
  });
  vi.stubGlobal("fetch", vi.fn(() => { throw new Error("Hosted API request forbidden"); }));
  host = document.createElement("div"); document.body.append(host); root = createRoot(host);
  await vi.runOnlyPendingTimersAsync();
  baselineTimers = vi.getTimerCount();
});

afterEach(async () => {
  await act(async () => { root?.unmount(); }); root = undefined;
  try {
    expect(fetch).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(baselineTimers);
  } finally {
    document.body.replaceChildren();
    vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers();
  }
});

/** Mount the public federation export, including React's double effect cycle. */
async function mount(): Promise<void> {
  await act(async () => { root?.render(<StrictMode><SnnStudioPanel /></StrictMode>); });
  expect(requests.map((request) => request.capability)).toEqual(["federation:config", "federation:config"]);
  expect(new Set(requests.map((request) => request.id)).size).toBe(2);
}

/** Answer the current request through the real DOM message event boundary. */
async function answer(result: unknown, ok: unknown = true): Promise<void> {
  const request = requests.at(-1);
  if (!request) throw new Error("Request missing");
  await act(async () => {
    window.dispatchEvent(new MessageEvent("message", { source: parent,
      data: { type: "scpn:bridge:response", id: request.id, ok, result, error: "private host detail" } }));
  });
}

const own = { studio: "sc-neurocore", version: "3.16.0", era: "v2", evidenceTypes: ["studio.model-scan.v1"] };

it("shows metadata while every workbench view remains unavailable, with no storage or sockets", async () => {
  const storage = vi.spyOn(Storage.prototype, "getItem");
  const socket = vi.spyOn(window, "WebSocket");
  await mount(); expect(host.textContent).toContain("Reading host catalogue"); await answer([own]);
  expect(host.querySelector("dl")?.textContent).toContain("sc-neurocoreVersion3.16.0Contract erav2");
  expect(host.textContent).toContain("studio.model-scan.v1");
  expect(host.querySelectorAll("li")).toHaveLength(25);
  for (const item of host.querySelectorAll("li")) expect(item.textContent).toContain("standalone workbench view not mounted here");
  expect(host.querySelector("button")?.textContent).toContain("Connect local runtime");
  expect(host.querySelector("input, select, textarea, canvas")).toBeNull();
  expect(storage).not.toHaveBeenCalled(); expect(socket).not.toHaveBeenCalled();
});

it("renders catalogue strings as text and distinguishes absent evidence", async () => {
  await mount(); await answer([{ ...own, version: "<script>private</script>", evidenceTypes: [] }]);
  expect(host.textContent).toContain("<script>private</script>");
  expect(host.querySelector("script")).toBeNull(); expect(host.textContent).toContain("None declared");
});

it.each([null, {}, "catalogue", [null], [[]], [false], [{}],
  [{ ...own, studio: "" }], [{ ...own, studio: 1 }],
  [{ ...own, version: "" }], [{ ...own, version: null }],
  [{ ...own, era: "" }], [{ ...own, era: [] }],
  [{ ...own, evidenceTypes: null }], [{ ...own, evidenceTypes: [3] }],
].map((result) => ({ result })))("refuses malformed catalogue $result", async ({ result }) => {
  await mount(); await answer(result);
  expect(host.querySelector('[role="alert"]')?.textContent).toBe("Host catalogue response is invalid.");
});

it.each([[], [{ ...own, studio: "other" }], [own, own]].map((result) => ({ result })))("requires a unique own entry in $result", async ({ result }) => {
  await mount(); await answer(result);
  expect(host.querySelector('[role="alert"]')?.textContent).toBe("Host catalogue does not uniquely identify SC-NeuroCore.");
});

it.each([false, null, "true"])("hides host refusal detail when ok=%j and retries", async (ok) => {
  await mount(); await answer(null, ok);
  expect(host.querySelector('[role="alert"]')?.textContent).toBe("Host catalogue request was refused.");
  expect(host.textContent).not.toContain("private host detail");
  await act(async () => { host.querySelector("button")?.click(); });
  expect(host.textContent).toContain("Reading host catalogue"); expect(requests).toHaveLength(3);
  await answer([own]); expect(host.querySelector("dl")).not.toBeNull();
});

it("ignores wrong source, id, type, malformed messages and cancelled StrictMode answers", async () => {
  await mount(); const active = requests.at(-1); const old = requests[0];
  if (!active || !old) throw new Error("Requests missing");
  const valid = { type: "scpn:bridge:response", id: active.id, ok: true, result: [own] };
  await act(async () => {
    window.dispatchEvent(new MessageEvent("message", { source: window, data: valid }));
    for (const data of [null, [], { ...valid, type: "other" }, { ...valid, id: "wrong" }, { ...valid, id: old.id }]) {
      window.dispatchEvent(new MessageEvent("message", { source: parent, data }));
    }
  });
  expect(host.textContent).toContain("Reading host catalogue");
  await answer([own]); expect(host.querySelector("dl")).not.toBeNull();
});

it("times out, ignores late answers and permits a fresh retry", async () => {
  await mount(); await act(async () => { await vi.advanceTimersByTimeAsync(5_000); });
  expect(host.querySelector('[role="alert"]')?.textContent).toBe("Host catalogue request timed out.");
  await answer([own]); expect(host.querySelector("dl")).toBeNull();
  await act(async () => { host.querySelector("button")?.click(); });
  await answer([own]); expect(host.querySelector("dl")).not.toBeNull();
});

it("cancels on unmount and discards late answers", async () => {
  await mount(); await act(async () => { root?.unmount(); }); root = undefined;
  await answer([own]); expect(host.childElementCount).toBe(0);
});

it("an already cancelled bridge caller sends nothing", async () => {
  const controller = new AbortController(); controller.abort();
  await expect(requestHostFederation(controller.signal)).rejects.toThrow("Host catalogue request cancelled.");
  expect(requests).toEqual([]);
});

it("reports a send failure with a fixed message and cleans up", async () => {
  vi.spyOn(parent, "postMessage").mockImplementation(() => { throw new Error("private failure"); });
  await act(async () => { root?.render(<SnnStudioPanel />); });
  expect(host.querySelector('[role="alert"]')?.textContent).toBe("Host catalogue request could not be sent.");
});


it("hides unexpected browser identity failures without sending requests or leaving timers", async () => {
  vi.spyOn(crypto, "randomUUID").mockImplementation(() => { throw new Error("private browser diagnostics"); });
  await act(async () => { root?.render(<SnnStudioPanel />); });
  expect(host.querySelector('[role="alert"]')?.textContent).toBe("Host catalogue is unavailable.");
  expect(host.textContent).not.toContain("private browser diagnostics");
  expect(requests).toEqual([]);
});


it("connects explicitly from catalogue metadata to the actual hosted scientific workspace", async () => {
  await mount(); await answer([own]);
  await act(async () => { host.querySelector("button")?.click(); });
  expect(requests.at(-1)?.capability).toBe("custom:sc-neurocore:api");
  expect(host.textContent).toContain("local runtime connection requested");
  await answer({ protocol: "sc-neurocore.host-api.v1", operations: ["simulate"] });
  expect(host.querySelector('[aria-label="Local scientific workspace"]')).not.toBeNull();
  expect(host.querySelector("textarea")).not.toBeNull();
  expect(host.querySelector("canvas")).toBeNull();
  expect(host.textContent).toContain("1 available actions");
});

it("does not restore catalogue state if unmounted immediately after its real message response", async () => {
  await mount(); const request = requests.at(-1); if (request === undefined) throw new Error("Missing catalogue request");
  await act(async () => {
    window.dispatchEvent(new MessageEvent("message", { source: parent,
      data: { type: "scpn:bridge:response", id: request.id, ok: true, result: [own] } }));
    root?.unmount(); root = undefined;
  });
  expect(host.childElementCount).toBe(0);
});
