// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — hosted client message and content contracts
// @vitest-environment happy-dom
// @vitest-environment-options {"settings":{"enableIframePageLoading":true}}

import { webcrypto } from "node:crypto";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { describeHost, downloadHostArtifact, hostedTextDigest, readHostArtifact,
  readHostJob, readHostJson, submitHostJob } from "./hostApi";
import { HOST_API_PROTOCOL, HOST_OPERATIONS } from "./hostApiTypes";

/** Closed request captured at the actual parent message boundary. */
interface Sent { id: string; capability: string; payload: unknown }
let parent: Window, controller: AbortController, sent: Sent[];
const JOB = "sj_1234567890abcdef";
const baseline = { jobId: JOB, kind: "analysis", status: "completed", artifacts: [] };

beforeEach(async () => {
  const frame = document.createElement("iframe"); document.body.append(frame);
  if (frame.contentWindow === null) throw new Error("Parent window unavailable");
  parent = frame.contentWindow; controller = new AbortController(); sent = [];
  vi.stubGlobal("parent", parent); vi.stubGlobal("crypto", webcrypto);
  vi.stubGlobal("fetch", vi.fn(() => { throw new Error("Hosted HTTP forbidden"); }));
  vi.spyOn(parent, "postMessage").mockImplementation((data: unknown) => {
    if (typeof data !== "object" || data === null || !("id" in data) || !("capability" in data)
      || !("payload" in data) || typeof data.id !== "string" || typeof data.capability !== "string") {
      throw new Error("Invalid outgoing request");
    }
    sent.push({ id: data.id, capability: data.capability, payload: data.payload });
  });
  await new Promise((resolve) => window.setTimeout(resolve, 0));
});
afterEach(() => {
  controller.abort(); expect(fetch).not.toHaveBeenCalled();
  document.body.replaceChildren(); vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers();
});

/** Send a correlated response from the real designated parent window. */
function answer(result: unknown, ok: unknown = true): void {
  const request = sent.at(-1); if (!request) throw new Error("No request sent");
  window.dispatchEvent(new MessageEvent("message", { source: parent,
    data: { type: "scpn:bridge:response", id: request.id, ok,
      result: typeof result === "object" && result !== null ? { protocol: HOST_API_PROTOCOL, ...result } : result,
      error: "private runtime diagnostics" } }));
}

it("describes only unique known installed operations using the exact custom capability", async () => {
  const pending = describeHost(controller.signal); answer({ operations: [...HOST_OPERATIONS] });
  expect(await pending).toEqual(HOST_OPERATIONS);
  expect(sent[0]).toMatchObject({ capability: "custom:sc-neurocore:api",
    payload: { protocol: HOST_API_PROTOCOL, action: "describe" } });
});

it.each([null, [], {}, { operations: null }, { operations: ["training"] },
  { operations: ["simulate", "simulate"] }, { operations: [1] }].map((value) => ({ value })))(
  "refuses malformed installed operations $value", async ({ value }) => {
    const pending = describeHost(controller.signal); answer(value);
    await expect(pending).rejects.toThrow(/invalid|refused/);
  },
);

it("retains original decimal, large-seed and declaration-order text on every operation", async () => {
  const json = '{"seed":18446744073709551615,"stream":4294967295,"value":1.0000000000000000001,"params":{"z":1,"a":2}}';
  for (const operation of HOST_OPERATIONS) {
    const pending = submitHostJob(operation, json, controller.signal);
    answer({ ...baseline, kind: operation.startsWith("ir.") ? "compiler" : "analysis" });
    expect((await pending).jobId).toBe(JOB);
    expect(sent.at(-1)?.payload).toEqual({ protocol: HOST_API_PROTOCOL, action: "submit", operation, json });
  }
});

it.each(["[]", "null", '"value"', "x", " ".repeat(1_048_577)])(
  "refuses a nonobject or oversized scientific document", async (json) => {
    await expect(submitHostJob("simulate", json, controller.signal)).rejects.toThrow();
    expect(sent).toEqual([]);
  },
);

it.each([
  { jobId: "other" }, { jobId: `${JOB}\n` }, { kind: "training" }, { status: "unknown-value" }, { artifacts: null },
  { artifacts: Array.from({ length: 1025 }, () => ({})) },
  { artifacts: [{ digest: "invalid", bytes: 1, result: true }] },
  { artifacts: [{ digest: `sha256:${"a".repeat(64)}\n`, bytes: 1, result: true }] },
  { artifacts: [{ digest: `sha256:${"a".repeat(64)}`, bytes: -1, result: true }] },
  { artifacts: [{ digest: `sha256:${"a".repeat(64)}`, bytes: 1.5, result: true }] },
  { artifacts: [{ digest: `sha256:${"a".repeat(64)}`, bytes: 1, result: "true" }] },
])("refuses malformed job projection %j", async (change) => {
  const pending = readHostJob(JOB, "status", controller.signal); answer({ ...baseline, ...change });
  await expect(pending).rejects.toThrow("invalid");
});

it("rejects crossed job identities and submitted worker families", async () => {
  let pending = readHostJob(JOB, "cancel", controller.signal);
  answer({ ...baseline, jobId: "sj_0000000000000000" });
  await expect(pending).rejects.toThrow("another job");
  pending = submitHostJob("ir.build", "{}", controller.signal); answer(baseline);
  await expect(pending).rejects.toThrow("family");
});

it.each(["models", "templates", "result"] as const)("verifies exact %s text, byte count and digest", async (action) => {
  const json = '{"seed":18446744073709551615,"label":"ž"}';
  const digest = await hostedTextDigest(json), bytes = new TextEncoder().encode(json).length;
  const pending = readHostJson(action === "result" ? { action, jobId: JOB } : { action }, controller.signal);
  answer({ json, digest, bytes }); expect(await pending).toEqual({ json, digest, bytes });
});

it.each(["digest", "size", "hash", "json", "bom", "bytes"])("refuses invalid %s result binding", async (mode) => {
  const json = mode === "json" ? "invalid JSON" : mode === "bom" ? "\ufeff{}" : "{}";
  const data = { json, digest: await hostedTextDigest(json), bytes: new TextEncoder().encode(json).length };
  if (mode === "digest") data.digest = "invalid";
  if (mode === "size") data.bytes += 1;
  if (mode === "hash") data.digest = `sha256:${"0".repeat(64)}`;
  const pending = readHostJson({ action: "result", jobId: JOB }, controller.signal);
  answer(mode === "bytes" ? { ...data, json: null } : data);
  await expect(pending).rejects.toThrow();
});

it("reads declared binary bytes independently of UTF-8 decoding", async () => {
  const bytes = new Uint8Array([0, 255, 128, 10]);
  const hash = new Uint8Array(await webcrypto.subtle.digest("SHA-256", bytes));
  const digest = `sha256:${Array.from(hash, (item) => item.toString(16).padStart(2, "0")).join("")}`;
  const pending = readHostArtifact(JOB, digest, controller.signal);
  answer({ encoding: "base64", data: btoa(String.fromCharCode(...bytes)), bytes: bytes.length, digest });
  expect(await pending).toEqual(bytes);
});

it.each(["encoding", "base64", "digest", "size"])("refuses invalid binary artifact %s", async (mode) => {
  const digest = await hostedTextDigest("artifact"), data = btoa("artifact");
  const pending = readHostArtifact(JOB, digest, controller.signal);
  answer({ encoding: mode === "encoding" ? "hex" : "base64", data: mode === "base64" ? "?" : data,
    digest: mode === "digest" ? `sha256:${"a".repeat(64)}` : digest, bytes: mode === "size" ? 999 : 8 });
  await expect(pending).rejects.toThrow();
});

it("requests a trusted parent download without sending content or a filename", async () => {
  const artifact = { digest: `sha256:${"b".repeat(64)}`, bytes: 19, result: false };
  const pending = downloadHostArtifact(JOB, artifact, controller.signal);
  answer({ downloaded: true, ...artifact }); await pending;
  expect(sent[0]?.payload).toEqual({ protocol: HOST_API_PROTOCOL, action: "download", jobId: JOB, digest: artifact.digest });
});

it.each(["status", "digest", "bytes"])("refuses a mismatched download %s receipt", async (mode) => {
  const artifact = { digest: `sha256:${"b".repeat(64)}`, bytes: 19, result: false };
  const pending = downloadHostArtifact(JOB, artifact, controller.signal);
  answer({ downloaded: mode !== "status", digest: mode === "digest" ? "other" : artifact.digest,
    bytes: mode === "bytes" ? 18 : 19 });
  await expect(pending).rejects.toThrow("receipt");
});

it("ignores other source, id and type, then cancels without accepting a late reply", async () => {
  const pending = describeHost(controller.signal), request = sent[0];
  if (!request) throw new Error("Missing request");
  for (const [source, data] of [[window, { type: "scpn:bridge:response", id: request.id }],
    [parent, null], [parent, { type: "other", id: request.id }], [parent, { type: "scpn:bridge:response", id: "other" }]] as const) {
    window.dispatchEvent(new MessageEvent("message", { source, data }));
  }
  controller.abort(); answer({ operations: [] }); await expect(pending).rejects.toThrow("cancelled");
  await expect(describeHost(controller.signal)).rejects.toThrow("cancelled");
  expect(sent).toHaveLength(1);
});

it("times out finitely and hides host diagnostic text on refusal", async () => {
  vi.useFakeTimers(); const pending = describeHost(controller.signal);
  const rejected = expect(pending).rejects.toThrow("timed out");
  await vi.advanceTimersByTimeAsync(45_000); await rejected; expect(vi.getTimerCount()).toBe(0);
  const refused = describeHost(controller.signal); answer(null, false);
  await expect(refused).rejects.toThrow("Local runtime request was refused.");
  expect(vi.getTimerCount()).toBe(0);
});

it("releases callbacks and uses fixed text when parent dispatch fails", async () => {
  vi.useFakeTimers(); vi.spyOn(parent, "postMessage").mockImplementation(() => { throw new Error("private"); });
  await expect(describeHost(controller.signal)).rejects.toThrow("Runtime request could not be sent.");
  expect(vi.getTimerCount()).toBe(0);
});
