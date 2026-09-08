// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Characterisation transport lifecycle tests

import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useStudioStore } from "./studio";

const initial = useStudioStore.getState();
const result = {
  fi_curve: { currents: [0, 1], rates: [0, 2] }, max_rate: 2,
  pattern: { pattern: "regular", description: "Regular" }, spike_count: 1,
  state_ranges: { v: { min: -70, max: -50, mean: -60 } },
  stats: { rate_hz: 2, isi_cv: null, isi_mean_ms: null, isi_histogram: null },
  threshold_current: 1, top_sensitivities: [],
};
let socket: Socket;

/**
 * Capture the browser-created socket for controlled delivery.
 *
 * @param created - New transport instance.
 */
function capture(created: Socket): void { socket = created; }

/** Controlled browser transport; production client and store remain real. */
class Socket {
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  send = vi.fn();
  close = vi.fn();
  constructor() { capture(this); }
}

beforeEach(() => {
  vi.stubGlobal("window", { location: { protocol: "http:", host: "localhost" } });
  vi.stubGlobal("WebSocket", Socket);
  useStudioStore.setState({ sourceMode: "model", selectedModelName: "lif" });
});
afterEach(() => { useStudioStore.setState(initial, true); vi.unstubAllGlobals(); });

/**
 * Deliver a server frame through the real progress parser.
 *
 * @param value - Wire payload.
 */
function frame(value: object): void { socket.onmessage?.({ data: JSON.stringify(value) }); }

it("marks socket success current and withdraws completion on rerun", () => {
  useStudioStore.getState().runCharacterize();
  socket.onopen?.();
  expect(JSON.parse(String(socket.send.mock.calls[0]?.[0]))).toMatchObject({ op: "characterize", config: { name: "lif" } });
  frame({ type: "complete", result });
  expect(useStudioStore.getState().analysisExperimentKey).not.toBeNull();
  expect(useStudioStore.getState().charResult).toEqual(result);
  expect(socket.close).toHaveBeenCalledOnce();
  useStudioStore.getState().runCharacterize();
  expect(useStudioStore.getState().analysisExperimentKey).toBeNull();
  frame({ type: "error", msg: "failed" });
  expect(useStudioStore.getState().charResult).toEqual(result);
  expect(useStudioStore.getState().isSimulating).toBe(false);
});

it.each(["complete", "error", "progress"])("discards stale %s frames", (type) => {
  useStudioStore.getState().runCharacterize();
  useStudioStore.getState().setSourceMode("ode");
  frame({ type, result, msg: "obsolete", pct: 50 });
  expect(useStudioStore.getState().charResult).toBeNull();
  expect(useStudioStore.getState().error).toBeNull();
  expect(useStudioStore.getState().progressMsg).toBe("");
  expect(useStudioStore.getState().isSimulating).toBe(false);
});

it("uses only one HTTP fallback and ignores subsequent socket frames", async () => {
  const fetch = vi.fn<typeof globalThis.fetch>().mockResolvedValue(new Response(JSON.stringify(result)));
  vi.stubGlobal("fetch", fetch);
  useStudioStore.getState().runCharacterize();
  const error = socket.onerror;
  error?.(); error?.();
  frame({ type: "error", msg: "obsolete socket" });
  await vi.waitFor(() => { expect(useStudioStore.getState().isSimulating).toBe(false); });
  expect(fetch).toHaveBeenCalledOnce();
  expect(useStudioStore.getState().error).toBeNull();
  expect(useStudioStore.getState().analysisExperimentKey).not.toBeNull();
});

it("falls back after a socket closes without a terminal frame", async () => {
  vi.stubGlobal("fetch", vi.fn<typeof globalThis.fetch>().mockResolvedValue(new Response(JSON.stringify(result))));
  useStudioStore.getState().runCharacterize();
  socket.onclose?.();
  await vi.waitFor(() => { expect(useStudioStore.getState().charResult).toEqual(result); });
});

it.each([false, true])("drops stale HTTP outcomes, rejected=%s", async (rejected) => {
  let finish = (): void => { throw new Error("no HTTP request"); };
  vi.stubGlobal("fetch", vi.fn<typeof globalThis.fetch>(() => new Promise<Response>((resolve, reject) => {
    finish = () => { if (rejected) reject(new Error("old failure")); else resolve(new Response(JSON.stringify(result))); };
  })));
  useStudioStore.getState().runCharacterize();
  socket.onerror?.();
  useStudioStore.getState().setSourceMode("ode");
  finish();
  await vi.waitFor(() => { expect(useStudioStore.getState().isSimulating).toBe(false); });
  expect(useStudioStore.getState().charResult).toBeNull();
  expect(useStudioStore.getState().error).toBeNull();
});

it("rejects malformed HTTP completion and reports transport failure", async () => {
  vi.stubGlobal("fetch", vi.fn<typeof globalThis.fetch>().mockResolvedValue(new Response("{}")));
  useStudioStore.getState().runCharacterize();
  socket.onerror?.();
  await vi.waitFor(() => { expect(useStudioStore.getState().error).toBe("Malformed characterisation result"); });
  vi.stubGlobal("fetch", vi.fn<typeof globalThis.fetch>().mockRejectedValue(new Error("offline")));
  useStudioStore.getState().runCharacterize();
  socket.onerror?.();
  await vi.waitFor(() => { expect(useStudioStore.getState().error).toContain("offline"); });
  expect(useStudioStore.getState().analysisExperimentKey).toBeNull();
});

it("handles invalid input and ignores a second invocation while busy", () => {
  useStudioStore.setState({ dt: Number.NaN });
  expect(() => { useStudioStore.getState().runCharacterize(); }).not.toThrow();
  expect(useStudioStore.getState().error).toContain("NaN");
  useStudioStore.setState({ dt: 0.1 });
  useStudioStore.getState().runCharacterize();
  const state = useStudioStore.getState();
  useStudioStore.getState().runCharacterize();
  expect(useStudioStore.getState()).toBe(state);
  frame({ type: "progress", pct: 50, msg: "half" });
  expect(useStudioStore.getState().progressPct).toBe(50);
  useStudioStore.setState({ dt: Number.NaN });
  frame({ type: "complete", result });
  expect(useStudioStore.getState().error).toContain("NaN");
  expect(useStudioStore.getState().isSimulating).toBe(false);
});
