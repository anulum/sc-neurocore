// @vitest-environment happy-dom
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio status bar names the run in progress

import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

import type { GraphSimResult, SimulateResponse } from "../api/types";
import { useStudioStore } from "../stores/studio";
import StatusBar from "./StatusBar";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** A finished single-model run, as the store holds it after "Run". */
const finishedRun: SimulateResponse = {
  time: [0.1, 100, 200],
  states: { v: [0, 3, 7] },
  current_trace: [1, 1, 2],
  spikes: [3],
  spike_count: 12,
  stats: { isi_cv: null, isi_histogram: null, isi_mean_ms: null, rate_hz: 60 },
  dt: 0.1,
  n_steps: 2000,
  run_metadata: {
    dt: 0.1,
    evidence_classification: "simulation",
    input_sha256: "1".repeat(64),
    n_steps: 2000,
    result_sha256: "2".repeat(64),
    sample_count: 3,
    schema_version: "studio.simulation-run.v2",
    source: "model",
    spike_count: 12,
    status: "completed",
    state_variables: ["v"],
  },
};

/** A finished network canvas run. */
const finishedGraph: GraphSimResult = { success: true, n_total: 40, n_spikes: 312, n_steps: 2000, dt: 0.1 };

/**
 * Render the status bar with the live store and hand its host to `inspect` while it is mounted.
 *
 * @param inspect - Reads the rendered bar.
 */
async function withStatusBar(inspect: (host: HTMLElement) => void): Promise<void> {
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  await act(async () => { root.render(<StatusBar />); });
  try {
    inspect(host);
  } finally {
    await act(async () => { root.unmount(); });
    host.remove();
  }
}

/**
 * Return the bar's one status region.
 *
 * @param host - The rendered bar's host.
 * @returns The region a screen reader announces.
 */
function statusRegion(host: HTMLElement): HTMLElement {
  const regions = host.querySelectorAll<HTMLElement>('[role="status"]');
  expect(regions).toHaveLength(1);
  const [region] = regions;
  if (region === undefined) throw new Error("The bar has no status region.");
  return region;
}

/**
 * Render the status bar with the live store and return its text.
 *
 * @returns The rendered text.
 */
async function statusText(): Promise<string> {
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  await act(async () => { root.render(<StatusBar />); });
  const text = container.textContent;
  await act(async () => { root.unmount(); });
  container.remove();
  return text;
}

describe("StatusBar", () => {
  const initial = useStudioStore.getState();
  afterEach(() => {
    useStudioStore.setState(initial, true);
  });

  it("names the run that holds the Studio instead of calling every run a simulation", async () => {
    useStudioStore.setState({ isSimulating: true, busyWith: "Network pipeline" });
    const pipeline = await statusText();
    expect(pipeline).toContain("Network pipeline running…");
    expect(pipeline).not.toContain("simulating");

    useStudioStore.setState({ isSimulating: true, busyWith: "Simulation" });
    expect(await statusText()).toContain("Simulation running…");
  });

  it("keeps its status region mounted before the first result, so that result is announced", async () => {
    useStudioStore.setState({ isSimulating: false, result: null, networkResult: null, graphSimResult: null });
    await withStatusBar((host) => {
      expect(statusRegion(host).textContent).toBe("");
      expect(host.textContent).toContain("? for shortcuts");
    });
  });

  it("puts a finished run in the status region, read as a list, and keeps the shortcut hint out of it", async () => {
    useStudioStore.setState({
      isSimulating: false, activeTab: "trace", sourceMode: "model", selectedModelName: "AdExNeuron",
      result: finishedRun, networkResult: null, graphSimResult: null,
    });
    await withStatusBar((host) => {
      const region = statusRegion(host);
      expect(region.textContent).toBe("AdExNeuron, 12 spikes, 60 Hz, dt=0.1 T=200ms");
      expect(region.textContent).not.toContain("shortcuts");
      // The separators are spoken, not drawn: the bar still shows the parts apart.
      for (const separator of region.querySelectorAll(".visually-hidden")) {
        expect(separator.textContent).toBe(", ");
      }
      expect(region.querySelectorAll(".visually-hidden")).toHaveLength(3);
    });
  });

  it("reports a finished network canvas run on the canvas, not the last single-model run", async () => {
    useStudioStore.setState({
      isSimulating: false, activeTab: "canvas", sourceMode: "model", selectedModelName: "AdExNeuron",
      result: finishedRun, networkResult: null, graphSimResult: finishedGraph,
    });
    await withStatusBar((host) => {
      expect(statusRegion(host).textContent).toBe("Network: 40 neurons, 312 spikes, 2000 steps × 0.1 ms");
    });
  });

  it("shows a count the network result does not carry as unknown, never as zero", async () => {
    useStudioStore.setState({
      isSimulating: false, activeTab: "canvas", result: null, networkResult: null, graphSimResult: { success: true },
    });
    await withStatusBar((host) => {
      expect(statusRegion(host).textContent).toBe("Network: ? neurons, ? spikes, ? steps × ? ms");
    });
  });

  it("reports a refused network canvas run, with its reasons, not the last single-model run", async () => {
    useStudioStore.setState({
      isSimulating: false, activeTab: "canvas", sourceMode: "model", selectedModelName: "AdExNeuron",
      result: finishedRun, networkResult: null,
      graphSimResult: { success: false, errors: ["Population 'Exc 0' has no neurons."] },
    });
    await withStatusBar((host) => {
      expect(statusRegion(host).textContent).toBe("Network run refused, Population 'Exc 0' has no neurons.");
    });
    useStudioStore.setState({ graphSimResult: { ...finishedGraph, success: false } });
    await withStatusBar((host) => {
      expect(statusRegion(host).textContent).toBe("Network run refused");
    });
  });

  it("reports the problems found in the network on the canvas, not an earlier run's counts", async () => {
    useStudioStore.setState({
      isSimulating: false, activeTab: "canvas", result: finishedRun, networkResult: null,
      graphSimResult: finishedGraph, graphErrors: ["Projection 'p0' has no source.", "Population 'Inh 1' is empty."],
    });
    await withStatusBar((host) => {
      expect(statusRegion(host).textContent).toBe("Network: 2 problems found");
    });
    useStudioStore.setState({ graphErrors: ["Projection 'p0' has no source."] });
    await withStatusBar((host) => {
      expect(statusRegion(host).textContent).toBe("Network: 1 problem found");
    });
  });

  it("reports nothing on the canvas before its first network run, not the last single-model run", async () => {
    useStudioStore.setState({
      isSimulating: false, activeTab: "canvas", sourceMode: "model", selectedModelName: "AdExNeuron",
      result: finishedRun, networkResult: null, graphSimResult: null, graphErrors: [],
    });
    await withStatusBar((host) => {
      expect(statusRegion(host).textContent).toBe("");
    });
  });

  it("reports a finished E-I network run on its own tab", async () => {
    useStudioStore.setState({
      isSimulating: false, activeTab: "network", result: finishedRun, graphSimResult: null,
      networkResult: {
        spike_times: [], spike_neurons: [], n_exc: 80, n_inh: 20, n_total: 100, n_spikes: 412,
        rate_time: [], exc_rates: [], inh_rates: [], duration: 200, dt: 0.1, mean_exc_rate: 5.2, mean_inh_rate: 11.4,
      },
    });
    await withStatusBar((host) => {
      expect(statusRegion(host).textContent).toBe("E-I Network: 80E/20I, 412 spikes, E: 5.2Hz, I: 11.4Hz");
    });
  });

  it("reports nothing on the E-I network tab before its first run, not the last single-model run", async () => {
    useStudioStore.setState({
      isSimulating: false, activeTab: "network", sourceMode: "model", selectedModelName: "AdExNeuron",
      result: finishedRun, networkResult: null, graphSimResult: finishedGraph,
    });
    await withStatusBar((host) => {
      expect(statusRegion(host).textContent).toBe("");
    });
  });

  it("wraps onto further lines in a narrow window instead of holding one fixed-height row", async () => {
    useStudioStore.setState({ isSimulating: false, activeTab: "trace", result: finishedRun, graphSimResult: null });
    await withStatusBar((host) => {
      const bar = host.firstElementChild as HTMLElement;
      expect(bar.style.flexWrap).toBe("wrap");
      expect(bar.style.height).toBe("");
      expect(bar.style.minHeight).toBe("18px");
      expect(statusRegion(host).style.flexWrap).toBe("wrap");
    });
  });
});
