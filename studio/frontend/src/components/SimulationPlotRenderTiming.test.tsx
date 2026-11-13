// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — the simulation plot's render timing mark, through the mounted plot
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { SimulateResponse } from "../api/client";
import { useStudioStore } from "../stores/studio";
import SimulationPlot from "./SimulationPlot";

const initial = useStudioStore.getState();
let root: Root | undefined;
let container: HTMLDivElement;

/**
 * Return a stand-in 2D context: every method does nothing, `measureText` has a width.
 *
 * The plot's drawing code is exercised elsewhere; these cases need only that a
 * drawing call succeeds, as it does in a browser.
 *
 * @returns The context.
 */
function standInContext(): CanvasRenderingContext2D {
  const fields: Record<string | symbol, unknown> = {
    measureText: () => ({ width: 10 }),
    createLinearGradient: () => ({ addColorStop: () => undefined }),
  };
  return new Proxy(fields, {
    get: (target, key) => (key in target ? target[key] : () => undefined),
    set: (target, key, value) => {
      target[key] = value;
      return true;
    },
  }) as unknown as CanvasRenderingContext2D;
}

/**
 * Build a simulation result of two samples.
 *
 * @param last - The second sample of `v`, so two results differ.
 * @returns The result.
 */
function simulation(last: number): SimulateResponse {
  return {
    current_trace: [1, 1],
    dt: 0.1,
    model_name: "custom",
    n_steps: 2,
    run_metadata: {
      dt: 0.1, evidence_classification: "simulation", input_sha256: "1".repeat(64), n_steps: 2,
      result_sha256: "2".repeat(64), sample_count: 2, schema_version: "studio.simulation-run.v1", source: "ode",
      spike_count: 0, status: "completed", state_variables: ["v"],
    },
    spike_count: 0,
    spikes: [],
    states: { v: [0, last] },
    stats: { isi_cv: null, isi_histogram: null, isi_mean_ms: null, rate_hz: 0 },
    time: [0, 0.1],
  };
}

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => standInContext());
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue(
    { width: 400, height: 300, top: 0, left: 0, right: 400, bottom: 300, x: 0, y: 0, toJSON: () => ({}) },
  );
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root?.unmount(); await Promise.resolve(); });
  useStudioStore.setState(initial, true);
  document.body.replaceChildren();
  performance.clearMarks();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

/** The trace-drawn marks recorded so far. */
function traceMarks(): number {
  return performance.getEntriesByName("studio:trace-drawn", "mark").length;
}

/**
 * Put a state into the store and let the plot draw it.
 *
 * @param state - The store fields of the case.
 */
async function show(state: Partial<ReturnType<typeof useStudioStore.getState>>): Promise<void> {
  await act(async () => {
    useStudioStore.setState(state);
    root?.render(<SimulationPlot />);
    await Promise.resolve();
  });
}

describe("SimulationPlot render timing", () => {
  it("marks each new result's trace once, and not a redraw of the same trace", async () => {
    await show({ activeTab: "trace", result: simulation(0.1) });
    expect(traceMarks()).toBe(1);
    await act(async () => { window.dispatchEvent(new Event("resize")); await Promise.resolve(); });
    expect(traceMarks()).toBe(1);
    await show({ result: simulation(0.2) });
    expect(traceMarks()).toBe(2);
  });

  it("marks nothing while another view, or no result, is drawn", async () => {
    await show({ activeTab: "trace", result: null });
    expect(traceMarks()).toBe(0);
    await show({ activeTab: "bifurcation", result: simulation(0.1) });
    expect(traceMarks()).toBe(0);
  });

  it("draws the same markup whether or not the timeline takes marks", async () => {
    await show({ activeTab: "trace", result: simulation(0.1) });
    const marked = container.innerHTML;
    await act(async () => { root?.unmount(); await Promise.resolve(); });
    root = createRoot(container);
    vi.spyOn(performance, "mark").mockImplementation(() => {
      throw new Error("no timeline");
    });
    await show({ activeTab: "trace", result: simulation(0.1) });
    expect(container.innerHTML).toBe(marked);
  });
});
