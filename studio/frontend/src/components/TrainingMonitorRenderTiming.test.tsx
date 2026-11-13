// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — the training charts' render timing mark, through the mounted monitor
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { TrainingEpochMetrics } from "../api/client";
import { useStudioStore } from "../stores/studio";
import TrainingMonitor from "./TrainingMonitor";

const pristine = useStudioStore.getState();

/**
 * Build one epoch's metrics.
 *
 * @param epoch - The epoch number.
 * @returns The metrics of that epoch.
 */
function epoch(epoch: number): TrainingEpochMetrics {
  return {
    epoch, train_loss: 1 / epoch, train_accuracy: 0.5, val_loss: 1 / epoch, val_accuracy: 0.5,
    layer_spike_rates: {}, param_snapshot: {},
  };
}

let root: Root | undefined;
let container: HTMLDivElement;

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("fetch", vi.fn(async () => new Response("[]", { headers: { "content-type": "application/json" } })));
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root?.unmount(); await Promise.resolve(); });
  useStudioStore.setState(pristine, true);
  document.body.replaceChildren();
  performance.clearMarks();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

/**
 * Render the monitor with a list of epochs.
 *
 * @param epochs - The epochs the store holds.
 */
async function show(epochs: TrainingEpochMetrics[]): Promise<void> {
  await act(async () => {
    useStudioStore.setState({ trainingEpochs: epochs });
    root?.render(<TrainingMonitor />);
    await Promise.resolve();
  });
}

/** The training-chart-drawn marks recorded so far. */
function chartMarks(): number {
  return performance.getEntriesByName("studio:training-chart-drawn", "mark").length;
}

describe("TrainingMonitor render timing", () => {
  it("marks the charts once per new epoch row, and nothing before the first", async () => {
    await show([]);
    expect(chartMarks()).toBe(0);
    await show([epoch(1)]);
    expect(chartMarks()).toBe(1);
    await show([epoch(1)]);
    expect(chartMarks()).toBe(1);
    await show([epoch(1), epoch(2)]);
    expect(chartMarks()).toBe(2);
  });

  it("draws the same markup whether or not the timeline takes marks", async () => {
    await show([epoch(1), epoch(2)]);
    const marked = container.innerHTML;
    await act(async () => { root?.unmount(); await Promise.resolve(); });
    root = createRoot(container);
    vi.spyOn(performance, "mark").mockImplementation(() => {
      throw new Error("no timeline");
    });
    await show([epoch(1), epoch(2)]);
    expect(container.innerHTML).toBe(marked);
  });
});
