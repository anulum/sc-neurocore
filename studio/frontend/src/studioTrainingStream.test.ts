// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio training stream event parsing tests

import { describe, expect, it } from "vitest";

import { parseStudioTrainingStreamMessage } from "./studioTrainingStream";

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
});
