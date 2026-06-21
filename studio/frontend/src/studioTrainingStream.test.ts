// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio training stream event parsing tests

import { describe, expect, it } from "vitest";

import type { TrainingEpochMetrics } from "./api/client";
import {
  connectStudioTrainingEventSource,
  parseStudioTrainingStreamMessage,
  studioTrainingStreamUrl,
  type StudioTrainingStreamEventSource,
  type StudioTrainingTerminalStatus,
} from "./studioTrainingStream";

class FakeTrainingEventSource implements StudioTrainingStreamEventSource {
  closed = false;
  onerror: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent<string>) => void) | null = null;

  close(): void {
    this.closed = true;
  }

  emit(data: unknown): void {
    this.onmessage?.(new MessageEvent("message", { data: JSON.stringify(data) }));
  }

  emitRaw(data: string): void {
    this.onmessage?.(new MessageEvent("message", { data }));
  }

  fail(): void {
    this.onerror?.(new Event("error"));
  }
}

describe("Studio training stream parser", () => {
  it("parses epoch metric events", () => {
    expect(parseStudioTrainingStreamMessage(JSON.stringify({
      event: "epoch",
      data: {
        epoch: 2,
        train_loss: 0.4,
        train_accuracy: 0.8,
        val_loss: 0.5,
        val_accuracy: 0.75,
        layer_spike_rates: { hidden: 0.12, bad: Number.NaN, text: "x" },
        param_snapshot: { beta: 0.9 },
      },
    }))).toEqual({
      kind: "epoch",
      metrics: {
        epoch: 2,
        train_loss: 0.4,
        train_accuracy: 0.8,
        val_loss: 0.5,
        val_accuracy: 0.75,
        layer_spike_rates: { hidden: 0.12 },
        param_snapshot: { beta: 0.9 },
      },
    });
  });

  it("parses terminal events", () => {
    expect(parseStudioTrainingStreamMessage(JSON.stringify({ event: "completed" }))).toEqual({
      kind: "terminal",
      status: "completed",
    });
    expect(parseStudioTrainingStreamMessage(JSON.stringify({ event: "stopped" }))).toEqual({
      kind: "terminal",
      status: "stopped",
    });
  });

  it("parses backend error events with a fallback message", () => {
    expect(parseStudioTrainingStreamMessage(JSON.stringify({
      event: "error",
      data: { message: "training diverged" },
    }))).toEqual({
      kind: "error",
      message: "training diverged",
    });
    expect(parseStudioTrainingStreamMessage(JSON.stringify({ event: "error", data: {} }))).toEqual({
      kind: "error",
      message: "Training failed",
    });
  });

  it("ignores malformed JSON and unsupported event names", () => {
    expect(parseStudioTrainingStreamMessage("{not-json")).toBeNull();
    expect(parseStudioTrainingStreamMessage(JSON.stringify({ event: "heartbeat" }))).toBeNull();
  });

  it("rejects incomplete or non-finite epoch metrics", () => {
    expect(parseStudioTrainingStreamMessage(JSON.stringify({
      event: "epoch",
      data: {
        epoch: 1,
        train_loss: Number.NaN,
        train_accuracy: 0.8,
        val_loss: 0.5,
        val_accuracy: 0.75,
      },
    }))).toBeNull();
    expect(parseStudioTrainingStreamMessage(JSON.stringify({
      event: "epoch",
      data: { epoch: 1 },
    }))).toBeNull();
  });

  it("builds an encoded training stream URL", () => {
    expect(studioTrainingStreamUrl("job/with spaces")).toBe(
      "/api/training/stream/job%2Fwith%20spaces",
    );
  });

  it("dispatches epoch updates without closing the stream", () => {
    const source = new FakeTrainingEventSource();
    const epochs: TrainingEpochMetrics[] = [];

    connectStudioTrainingEventSource("job-1", {
      onDisconnected: () => { throw new Error("unexpected disconnect"); },
      onEpoch: (metrics) => epochs.push(metrics),
      onError: () => { throw new Error("unexpected error"); },
      onTerminal: () => { throw new Error("unexpected terminal"); },
    }, () => source);

    source.emit({
      event: "epoch",
      data: {
        epoch: 3,
        train_loss: 0.3,
        train_accuracy: 0.9,
        val_loss: 0.35,
        val_accuracy: 0.88,
      },
    });

    expect(epochs).toEqual([{
      epoch: 3,
      layer_spike_rates: {},
      param_snapshot: {},
      train_accuracy: 0.9,
      train_loss: 0.3,
      val_accuracy: 0.88,
      val_loss: 0.35,
    }]);
    expect(source.closed).toBe(false);
  });

  it("dispatches terminal updates and closes the stream", () => {
    const source = new FakeTrainingEventSource();
    const terminals: StudioTrainingTerminalStatus[] = [];

    connectStudioTrainingEventSource("job-1", {
      onDisconnected: () => { throw new Error("unexpected disconnect"); },
      onEpoch: () => { throw new Error("unexpected epoch"); },
      onError: () => { throw new Error("unexpected error"); },
      onTerminal: (status) => terminals.push(status),
    }, () => source);

    source.emit({ event: "completed" });

    expect(terminals).toEqual(["completed"]);
    expect(source.closed).toBe(true);
  });

  it("dispatches backend errors and closes the stream", () => {
    const source = new FakeTrainingEventSource();
    const errors: string[] = [];

    connectStudioTrainingEventSource("job-1", {
      onDisconnected: () => { throw new Error("unexpected disconnect"); },
      onEpoch: () => { throw new Error("unexpected epoch"); },
      onError: (message) => errors.push(message),
      onTerminal: () => { throw new Error("unexpected terminal"); },
    }, () => source);

    source.emit({ event: "error", data: { message: "training diverged" } });

    expect(errors).toEqual(["training diverged"]);
    expect(source.closed).toBe(true);
  });

  it("dispatches stream disconnects and closes the stream", () => {
    const source = new FakeTrainingEventSource();
    let disconnected = false;

    connectStudioTrainingEventSource("job-1", {
      onDisconnected: () => { disconnected = true; },
      onEpoch: () => { throw new Error("unexpected epoch"); },
      onError: () => { throw new Error("unexpected error"); },
      onTerminal: () => { throw new Error("unexpected terminal"); },
    }, () => source);

    source.fail();

    expect(disconnected).toBe(true);
    expect(source.closed).toBe(true);
  });

  it("ignores malformed stream messages without closing", () => {
    const source = new FakeTrainingEventSource();
    let callbackCount = 0;

    connectStudioTrainingEventSource("job-1", {
      onDisconnected: () => { callbackCount += 1; },
      onEpoch: () => { callbackCount += 1; },
      onError: () => { callbackCount += 1; },
      onTerminal: () => { callbackCount += 1; },
    }, () => source);

    source.emitRaw("{not-json");
    source.emit({ event: "heartbeat" });

    expect(callbackCount).toBe(0);
    expect(source.closed).toBe(false);
  });
});
