// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — the network view's render timing mark, through the mounted canvas
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { PopulationNode } from "../api/client";
import { useStudioStore } from "../stores/studio";
import NetworkCanvas from "./NetworkCanvas";

const pristine = useStudioStore.getState();

const POPULATIONS: PopulationNode[] = [
  { count: 100, id: "p1", label: "Input", model: "LIF", neuron_type: "excitatory", params: {}, position: { x: 0, y: 0 }, type: "population" },
  { count: 20, id: "p2", label: "Output", model: "LIF", neuron_type: "inhibitory", params: {}, position: { x: 200, y: 0 }, type: "population" },
];

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

/** Render the network view. */
async function mount(): Promise<void> {
  await act(async () => {
    root?.render(<NetworkCanvas />);
    await Promise.resolve();
  });
}

/** The network-drawn marks recorded so far. */
function networkMarks(): number {
  return performance.getEntriesByName("studio:network-drawn", "mark").length;
}

describe("NetworkCanvas render timing", () => {
  it("marks nothing for an empty graph", async () => {
    useStudioStore.setState({ graphPopulations: [], graphProjections: [] });
    await mount();
    expect(networkMarks()).toBe(0);
  });

  it("marks each time the view draws a changed graph or switches between canvas and table", async () => {
    useStudioStore.setState({ graphPopulations: POPULATIONS, graphProjections: [] });
    await mount();
    expect(networkMarks()).toBe(1);
    await mount();
    expect(networkMarks()).toBe(1);
    await act(async () => {
      useStudioStore.getState().updatePopulation("p2", { position: { x: 240, y: 10 } });
      await Promise.resolve();
    });
    expect(networkMarks()).toBe(2);
    const toggle = container.querySelector<HTMLButtonElement>("button[aria-pressed]");
    await act(async () => { toggle?.click(); await Promise.resolve(); });
    expect(networkMarks()).toBe(3);
  });

  it("draws the same markup whether or not the timeline takes marks", async () => {
    useStudioStore.setState({ graphPopulations: POPULATIONS, graphProjections: [] });
    await mount();
    const marked = container.innerHTML;
    await act(async () => { root?.unmount(); await Promise.resolve(); });
    root = createRoot(container);
    vi.spyOn(performance, "mark").mockImplementation(() => {
      throw new Error("no timeline");
    });
    await mount();
    expect(container.innerHTML).toBe(marked);
  });
});
