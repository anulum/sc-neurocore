// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — authenticated Training Monitor SSE transport tests

import { afterEach, describe, expect, it, vi } from "vitest";

import { setStudioAuthToken } from "./api/http";
import { createFetchStudioTrainingSource } from "./studioTrainingFetchSource";
import { connectStudioTrainingEventSource } from "./studioTrainingStream";

const encoder = new TextEncoder();

afterEach(() => {
  setStudioAuthToken(null);
  vi.unstubAllGlobals();
});

describe("authenticated Training Monitor stream", () => {
  it("sends the bearer header and decodes split SSE frames through the public connector", async () => {
    setStudioAuthToken("test-session-token");
    const fetcher = vi.fn(async (_url: string | URL | Request, _init?: RequestInit) =>
      new Response(new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode('data: {"event":"epoch","data":{"epoch":1,"train_loss":0.2,'));
          controller.enqueue(encoder.encode('"train_accuracy":0.8,"val_loss":0.3,"val_accuracy":0.7}}\r\n\r'));
          controller.enqueue(encoder.encode('\ndata: {"event":"interrupted"}\n\n'));
          controller.close();
        },
      }), { headers: { "content-type": "text/event-stream; charset=utf-8" } }));
    const epochs: number[] = [];
    const terminals: string[] = [];
    const disconnected = vi.fn();
    vi.stubGlobal("fetch", fetcher);

    const source = connectStudioTrainingEventSource("sj_training", {
      onDisconnected: disconnected,
      onEpoch: (metrics) => epochs.push(metrics.epoch),
      onError: () => { throw new Error("unexpected training error"); },
      onTerminal: (status) => terminals.push(status),
    });

    await vi.waitFor(() => { expect(terminals).toEqual(["interrupted"]); });
    expect(epochs).toEqual([1]);
    expect(disconnected).not.toHaveBeenCalled();
    const [url, init] = fetcher.mock.calls[0] ?? [];
    expect(url).toBe("/api/training/stream/sj_training");
    expect(init?.headers).toEqual({
      Accept: "text/event-stream",
      Authorization: "Bearer test-session-token",
    });
    expect(init?.cache).toBe("no-store");
    expect(init?.redirect).toBe("error");
    expect(init?.signal?.aborted).toBe(true);
    source.close();
  });

  it("reports an unauthorised HTTP response as a disconnected stream", async () => {
    setStudioAuthToken("test-session-token");
    const fetcher = vi.fn(async (_url: string | URL | Request, _init?: RequestInit) =>
      new Response(null, { status: 401 }));
    const disconnected = vi.fn();
    const source = connectStudioTrainingEventSource("sj_training", {
      onDisconnected: disconnected,
      onEpoch: () => { throw new Error("unexpected epoch"); },
      onError: () => { throw new Error("unexpected job error"); },
      onTerminal: () => { throw new Error("unexpected terminal"); },
    }, (url) => createFetchStudioTrainingSource(url, fetcher));

    await vi.waitFor(() => { expect(disconnected).toHaveBeenCalledTimes(1); });
    expect(fetcher.mock.calls[0]?.[0]).toBe("/api/training/stream/sj_training");
    source.close();
  });

  it("refuses a cross-origin path before sending a bearer token", () => {
    setStudioAuthToken("test-session-token");
    const fetcher = vi.fn(async () => new Response(null, { status: 200 }));

    expect(() => createFetchStudioTrainingSource(
      "https://other.example/api/training/stream/sj_training",
      fetcher,
    )).toThrow("Training event stream path is invalid");
    expect(fetcher).not.toHaveBeenCalled();
  });

  it("aborts an unfinished stream without delivering a late frame", async () => {
    const fetcher = vi.fn(async (_url: string | URL | Request, _init?: RequestInit) =>
      new Response(new ReadableStream<Uint8Array>(), {
        headers: { "content-type": "text/event-stream" },
      }));
    const terminal = vi.fn();
    const disconnected = vi.fn();
    const source = connectStudioTrainingEventSource("sj_training", {
      onDisconnected: disconnected,
      onEpoch: () => undefined,
      onError: () => undefined,
      onTerminal: terminal,
    }, (url) => createFetchStudioTrainingSource(url, fetcher));
    await vi.waitFor(() => { expect(fetcher).toHaveBeenCalledTimes(1); });
    source.close();
    await Promise.resolve();

    expect(fetcher.mock.calls[0]?.[1]?.signal?.aborted).toBe(true);
    expect(terminal).not.toHaveBeenCalled();
    expect(disconnected).not.toHaveBeenCalled();
  });
});
