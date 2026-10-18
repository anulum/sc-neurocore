// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Progress client JSON and protocol binding tests

import { afterEach, expect, it, vi } from "vitest";
import { connectProgress } from "./progressApi";
import { setStudioAuthToken } from "./http";
import type { ProgressMessage } from "./types";

/** Captures transport calls while exercising the production progress client and JSON codec. */
class RecordingSocket {
  static instances: RecordingSocket[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  sent: string[] = [];

  /** Record the URL and protocols the production client selected. */
  constructor(readonly url: string, readonly protocols?: string[]) {
    RecordingSocket.instances.push(this);
  }

  /** Retain the exact frame emitted by the production serializer. */
  send(text: string): void {
    this.sent.push(text);
  }
}

afterEach(() => {
  setStudioAuthToken(null);
  RecordingSocket.instances = [];
  vi.unstubAllGlobals();
});

it("binds exact integer seed frames and parses responses through the production client", () => {
  vi.stubGlobal("window", { location: { protocol: "https:", host: "studio.local" } });
  vi.stubGlobal("WebSocket", RecordingSocket);
  setStudioAuthToken("unit-protocol");
  const receive = vi.fn<(message: ProgressMessage) => void>();
  const opened = connectProgress("characterize", { params: { seed: 9223372036854775807n } }, receive);
  const recorded = RecordingSocket.instances.at(-1);
  if (!recorded) throw new Error("The progress client did not open its transport.");
  expect(opened).toBe(recorded);
  expect(recorded.url).toBe("wss://studio.local/ws/progress");
  expect(recorded.protocols).toEqual(["studio-auth", "studio-bearer.unit-protocol"]);
  recorded.onopen?.();
  expect(recorded.sent).toEqual(['{"op":"characterize","config":{"params":{"seed":9223372036854775807}}}']);
  recorded.onmessage?.({ data: '{"type":"complete","result":{"seed":9223372036854775807}}' });
  expect(receive).toHaveBeenLastCalledWith({ type: "complete", result: { seed: 9223372036854775807n } });
  recorded.onmessage?.({ data: "{" });
  expect(receive).toHaveBeenCalledTimes(1);
  recorded.onerror?.();
  expect(receive).toHaveBeenLastCalledWith({ type: "error", msg: "WebSocket connection failed" });
});

it("refuses unsupported exact integer output before opening a progress transport", () => {
  const descriptor = Object.getOwnPropertyDescriptor(JSON, "rawJSON");
  if (!descriptor) throw new Error("The test runtime lacks raw JSON support.");
  vi.stubGlobal("window", { location: { protocol: "http:", host: "studio.local" } });
  vi.stubGlobal("WebSocket", RecordingSocket);
  try {
    Reflect.set(JSON, "rawJSON", undefined);
    expect(() => connectProgress("characterize", { seed: 9223372036854775807n }, vi.fn()))
      .toThrow("cannot write large Studio seeds");
    expect(RecordingSocket.instances).toHaveLength(0);
  } finally {
    Object.defineProperty(JSON, "rawJSON", descriptor);
  }
});
