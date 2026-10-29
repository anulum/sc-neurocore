// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — hosted workspace DOM and message integration
// @vitest-environment happy-dom
// @vitest-environment-options {"settings":{"enableIframePageLoading":true}}

import { createHash, webcrypto } from "node:crypto";
import { act, StrictMode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import HostedExecutionPanel from "./HostedExecutionPanel";
import { HOST_API_PROTOCOL, HOST_OPERATIONS, type HostOperation } from "./hostApiTypes";
import { hostedExample, HOST_OPERATION_LABELS } from "./hostedExamples";

/** Explicit parent contract fixture; installed execution is verified separately. */
interface Request { id: string; payload: Record<string, unknown>; capability: string }
let root: Root | null, host: HTMLDivElement, parent: Window, requests: Request[];
let status: string, refused: boolean, hold: boolean, operation: string;
let missingGraph: boolean, malformedResult: boolean;
const JOB = "sj_1234567890abcdef";

/** Compute a real content binding for a UI contract document. */
function document(json: string): { json: string; digest: string; bytes: number } {
  return { json, digest: `sha256:${createHash("sha256").update(json).digest("hex")}`,
    bytes: new TextEncoder().encode(json).length };
}
/** Produce only the explicit UI contract fixture, without a scientific runtime claim. */
function result(): ReturnType<typeof document> {
  return document(operation.startsWith("ir.") ? JSON.stringify({ accepted: true, refusal: null,
    result: { ir_text: "native graph contract fixture", valid: true, source_replay: "passed",
      systemverilog: operation === "ir.emit-sv" ? "module fixture; endmodule" : null } })
    : '{"voltage":[0,1],"seed":18446744073709551615}');
}
/** Project finite digest-only artifacts exactly as the real host does. */
function artifacts(): { digest: string; bytes: number; result: boolean }[] {
  if (status !== "completed") return [];
  const selected = result();
  return [{ digest: selected.digest, bytes: selected.bytes, result: true },
    ...(operation.startsWith("ir.") && !missingGraph ? [{ ...document("native graph contract fixture"), result: false }] : []),
    ...(operation === "ir.emit-sv" ? [{ ...document("module fixture; endmodule"), result: false }] : [])];
}
/** Return the current fixture projection independently from request arguments. */
function job(): object {
  return { jobId: JOB, kind: operation.startsWith("ir.") ? "compiler" : "analysis", status, artifacts: artifacts() };
}

beforeEach(async () => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true); vi.stubGlobal("crypto", webcrypto);
  vi.stubGlobal("fetch", vi.fn(() => { throw new Error("Child backend HTTP forbidden"); }));
  const frame = window.document.createElement("iframe"); window.document.body.append(frame);
  if (frame.contentWindow === null) throw new Error("Parent missing");
  parent = frame.contentWindow; vi.stubGlobal("parent", parent);
  requests = []; status = "pending"; refused = false; hold = false; operation = "simulate"; missingGraph = false; malformedResult = false;
  vi.spyOn(parent, "postMessage").mockImplementation((data: unknown) => {
    if (typeof data !== "object" || data === null || !("id" in data) || typeof data.id !== "string"
      || !("payload" in data) || typeof data.payload !== "object" || data.payload === null
      || !("capability" in data) || data.capability !== "custom:sc-neurocore:api") throw new Error("Invalid bridge request");
    const request: Request = { id: data.id, capability: data.capability, payload: { ...data.payload } };
    requests.push(request);
    if (hold) return;
    const payload = request.payload;
    let response: object;
    switch (payload.action) {
      case "describe": response = { operations: [...HOST_OPERATIONS] }; break;
      case "submit": operation = String(payload.operation); response = job(); break;
      case "status": response = job(); break;
      case "cancel": status = "cancelled"; response = job(); break;
      case "result": response = malformedResult ? document('{"accepted":true}') : result(); break;
      case "models": response = document('{"models":["lif"]}'); break;
      case "templates": response = document('{"equations":["dv/dt=-v"]}'); break;
      case "download": {
        const selected = artifacts().find((item) => item.digest === payload.digest);
        if (!selected) throw new Error("UI requested undeclared download");
        response = { downloaded: true, digest: selected.digest, bytes: selected.bytes }; break;
      }
      default: throw new Error("Unknown UI action");
    }
    queueMicrotask(() => { window.dispatchEvent(new MessageEvent("message", { source: parent,
      data: { type: "scpn:bridge:response", id: request.id, ok: !refused,
        result: { protocol: HOST_API_PROTOCOL, ...response }, error: "private backend detail" } })); });
  });
  host = window.document.createElement("div"); window.document.body.append(host); root = createRoot(host);
  await new Promise((resolve) => window.setTimeout(resolve, 0));
});
afterEach(async () => {
  await act(async () => { root?.unmount(); }); root = null;
  expect(fetch).not.toHaveBeenCalled(); window.document.body.replaceChildren();
  vi.restoreAllMocks(); vi.unstubAllGlobals();
});

/** Mount the actual component and settle its real parent-message effects. */
async function mount(): Promise<void> {
  await act(async () => { root?.render(<StrictMode><HostedExecutionPanel /></StrictMode>); });
}
/** Read a user-visible button without relying on an internal component state. */
function button(label: string): HTMLButtonElement {
  const selected = Array.from(host.querySelectorAll("button")).find((item) => item.textContent === label);
  if (!selected) throw new Error(`Missing button ${label}`);
  return selected;
}
/** Dispatch an actual DOM click then allow asynchronous message/hash work to settle. */
async function click(label: string): Promise<void> {
  expect(button(label).disabled).toBe(false);
  await act(async () => { button(label).click(); });
  const deadline = Date.now() + 2_000;
  while (host.textContent.includes("Waiting for local runtime…")) {
    expect(Date.now()).toBeLessThan(deadline);
    await act(async () => { await new Promise((resolve) => window.setTimeout(resolve, 1)); });
  }
}
/** Select a domain action through the actual controlled DOM input. */
async function choose(selected: HostOperation): Promise<void> {
  const control = host.querySelector("select"); if (!control) throw new Error("Missing action select");
  await act(async () => { control.value = selected; control.dispatchEvent(new Event("change", { bubbles: true })); });
  await click("Load example / last graph");
}

it.each(HOST_OPERATIONS)("runs reachable %s editor/job/result workflow through the public client", async (selected) => {
  await mount(); expect(host.textContent).toContain("9 available actions");
  await choose(selected); status = "completed";
  await click("Run scientific action");
  expect(requests.find((request) => request.payload.action === "submit")?.payload).toEqual({
    protocol: HOST_API_PROTOCOL, action: "submit", operation: selected, json: hostedExample(selected, ""),
  });
  expect(host.querySelector('[aria-label="Scientific result"]')?.textContent).toContain(result().json);
  await click("Download result JSON");
  expect(requests.at(-1)?.payload).toEqual({ protocol: HOST_API_PROTOCOL, action: "download", jobId: JOB, digest: result().digest });
  if (selected.startsWith("ir.")) {
    await click("Download native graph");
    if (selected === "ir.emit-sv") await click("Download SystemVerilog");
    await choose("ir.replay");
    expect(host.querySelector("textarea")?.value).toBe(hostedExample("ir.replay", "native graph contract fixture"));
  }
  expect(host.textContent).toContain(HOST_OPERATION_LABELS[selected]);
});

it("keeps an active job visible, polls explicitly, cancels, and permits a new action only after settlement", async () => {
  await mount(); await click("Run scientific action");
  expect(button("Run scientific action").disabled).toBe(true);
  expect(host.querySelector('[aria-label="Current scientific job"]')?.textContent).toContain("pending");
  status = "running"; await click("Refresh job and result");
  expect(host.querySelector('[aria-label="Current scientific job"]')?.textContent).toContain("running");
  await click("Cancel job"); expect(button("Cancel job").disabled).toBe(true);
  expect(button("Run scientific action").disabled).toBe(false);
  expect(host.textContent).toContain("This job produced no completed result");
  expect(host.querySelector('[aria-label="Scientific result"]')).toBeNull();
});

it("reads finite model and template catalogues without mounting the standalone store", async () => {
  await mount(); await click("Browse model catalogue"); expect(host.textContent).toContain('{"models":["lif"]}');
  await click("Browse equation templates"); expect(host.textContent).toContain('dv/dt=-v');
});

it("keeps backend refusals explicit and permits an actual connection retry", async () => {
  refused = true; await mount(); expect(host.textContent).toContain("Local runtime connection unavailable");
  expect(host.textContent).not.toContain("private backend detail");
  refused = false; await click("Retry runtime connection"); expect(host.textContent).toContain("9 available actions");
});

it("aborts outstanding messages on unmount without a child HTTP request or socket", async () => {
  hold = true; const sockets = vi.spyOn(window, "WebSocket");
  await mount(); expect(host.textContent).toContain("Checking installed local runtime");
  await act(async () => { root?.unmount(); }); root = null;
  expect(host.childElementCount).toBe(0); expect(sockets).not.toHaveBeenCalled();
  for (const request of requests) window.dispatchEvent(new MessageEvent("message", { source: parent,
    data: { type: "scpn:bridge:response", id: request.id, ok: true,
      result: { protocol: HOST_API_PROTOCOL, operations: HOST_OPERATIONS } } }));
  expect(host.childElementCount).toBe(0);
});


it("edits original numeric text, rejects malformed JSON safely, and allows correcting the request", async () => {
  await mount();
  const input = host.querySelector("textarea"); if (input === null) throw new Error("Missing scientific editor");
  const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set?.bind(input);
  if (setter === undefined) throw new Error("Missing browser textarea setter");
  await act(async () => { setter("{"); input.dispatchEvent(new Event("input", { bubbles: true })); });
  await click("Run scientific action");
  expect(host.querySelector('[role="alert"]')?.textContent).toBe("Runtime operation could not be completed.");
  expect(requests.some((item) => item.payload.action === "submit")).toBe(false);
  const json = '{"seed":18446744073709551615,"value":1.0000000000000000001}';
  await act(async () => { setter(json); input.dispatchEvent(new Event("input", { bubbles: true })); });
  await click("Run scientific action");
  expect(requests.at(-1)?.payload.json).toBe(json);
});

it("shows a fixed host refusal without discarding a pending job or exposing diagnostics", async () => {
  await mount(); await click("Run scientific action"); refused = true;
  await click("Refresh job and result");
  expect(host.querySelector('[role="alert"]')?.textContent).toBe("Local runtime request was refused.");
  expect(host.textContent).toContain("pending"); expect(host.textContent).not.toContain("private backend detail");
  refused = false; status = "completed"; await click("Refresh job and result");
  await click("Download artifact 1"); expect(requests.at(-1)?.payload.action).toBe("download");
});

it("refuses malformed compiler results and undeclared graph downloads through the visible UI", async () => {
  await mount(); await choose("ir.build"); status = "completed"; malformedResult = true;
  await click("Run scientific action");
  expect(host.querySelector('[role="alert"]')?.textContent).toBe("Compiler outcome is invalid.");
  expect(host.querySelector('[aria-label="Scientific result"]')).toBeNull();
  malformedResult = false; missingGraph = true; await click("Refresh job and result");
  const before = requests.filter((item) => item.payload.action === "download").length;
  await click("Download native graph");
  expect(host.querySelector('[role="alert"]')?.textContent).toBe("The selected result has no declared downloadable artifact.");
  expect(requests.filter((item) => item.payload.action === "download")).toHaveLength(before);
});

it("ignores an unavailable select option rather than changing the scientific action", async () => {
  await mount(); const select = host.querySelector("select"); if (select === null) throw new Error("Missing action");
  await act(async () => { select.value = "other"; select.dispatchEvent(new Event("change", { bubbles: true })); });
  await click("Load example / last graph");
  expect(host.querySelector("textarea")?.value).toBe(hostedExample("simulate", ""));
});

it.each(["describe", "submit", "status", "cancel", "result", "models", "templates", "retry"])(
  "unmounts after an actual %s response without reviving the workspace",
  async (action) => {
    if (action === "retry") { refused = true; await mount(); refused = false; }
    else if (action !== "describe") await mount();
    if (["status", "cancel", "result"].includes(action)) await click("Run scientific action");
    hold = true;
    await act(async () => {
      if (action === "describe") root?.render(<StrictMode><HostedExecutionPanel /></StrictMode>);
      else button(({ submit: "Run scientific action", status: "Refresh job and result", cancel: "Cancel job",
        result: "Refresh job and result", models: "Browse model catalogue", templates: "Browse equation templates",
        retry: "Retry runtime connection" } as Record<string, string>)[action] ?? "missing").click();
    });
    const selected = requests.at(-1); if (selected === undefined) throw new Error("Missing lifecycle request");
    if (action === "result") {
      await act(async () => { window.dispatchEvent(new MessageEvent("message", { source: parent,
        data: { type: "scpn:bridge:response", id: selected.id, ok: true,
          result: { protocol: HOST_API_PROTOCOL, jobId: JOB, kind: "analysis", status: "completed", artifacts: [] } } })); });
    }
    const pending = requests.at(-1); if (pending === undefined) throw new Error("Missing response identity");
    const response = ["describe", "retry"].includes(action) ? { operations: [...HOST_OPERATIONS] }
      : ["result", "models", "templates"].includes(action) ? result() : job();
    await act(async () => {
      window.dispatchEvent(new MessageEvent("message", { source: parent,
        data: { type: "scpn:bridge:response", id: pending.id, ok: true, result: { protocol: HOST_API_PROTOCOL, ...response } } }));
      root?.unmount(); root = null;
    });
    expect(host.childElementCount).toBe(0);
  },
);


it("cancels a pending user submission on unmount and leaves no visible workspace", async () => {
  await mount(); hold = true;
  await act(async () => { button("Run scientific action").click(); });
  expect(requests.at(-1)?.payload.action).toBe("submit");
  expect(host.textContent).toContain("Waiting for local runtime");
  await act(async () => { root?.unmount(); }); root = null;
  expect(host.childElementCount).toBe(0);
});
