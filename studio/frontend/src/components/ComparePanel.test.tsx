// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — the A/B compare panel, mounted with its store
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import ComparePanel from "./ComparePanel";

const store = vi.hoisted(() => ({
  activeTab: "trace",
  models: [] as { name: string }[],
  selectedModelName: "LIFNeuron",
  duration: 500,
  current: 12.5,
  protocol: { kind: "step" },
  runCompare: vi.fn(async () => undefined),
  isSimulating: false,
}));

vi.mock("../stores/studio", () => ({
  useStudioStore: () => store,
}));

let root: Root | undefined;
let container: HTMLDivElement;

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  store.activeTab = "trace";
  store.models = [{ name: "LIFNeuron" }, { name: "AdExNeuron" }, { name: "Izhikevich2007Neuron" }];
  store.selectedModelName = "LIFNeuron";
  store.isSimulating = false;
  store.runCompare.mockClear();
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root?.unmount(); await Promise.resolve(); });
  document.body.replaceChildren();
  vi.unstubAllGlobals();
});

/** Mount the panel. */
async function mount(): Promise<void> {
  await act(async () => { root?.render(<ComparePanel />); await Promise.resolve(); });
}

/** The model B picker. */
function picker(): HTMLSelectElement {
  const select = container.querySelector<HTMLSelectElement>('select[aria-label="Model B"]');
  if (select === null) throw new Error("the panel drew no model B picker");
  return select;
}

/** The run button. */
function runButton(): HTMLButtonElement {
  const button = [...container.querySelectorAll("button")].find((item) => item.textContent === "Run A/B");
  if (button === undefined) throw new Error("the panel drew no run button");
  return button;
}

/**
 * Choose model B through the change event.
 *
 * @param value - The model's name.
 */
async function chooseModelB(value: string): Promise<void> {
  await act(async () => {
    picker().value = value;
    picker().dispatchEvent(new Event("change", { bubbles: true }));
    await Promise.resolve();
  });
}

describe("ComparePanel", () => {
  it("shows itself only in the trace and compare views", async () => {
    for (const [tab, shown] of [["trace", true], ["compare", true], ["canvas", false], ["train", false]] as const) {
      store.activeTab = tab;
      await mount();
      expect(container.querySelector("h2")?.textContent === "A/B compare").toBe(shown);
    }
  });

  it("offers every other model as model B, never the selected one", async () => {
    await mount();
    expect([...picker().options].map((option) => option.value)).toEqual(["", "AdExNeuron", "Izhikevich2007Neuron"]);
    expect(runButton().title).toBe("Run model B beside LIFNeuron on the same drive");
  });

  it("keeps the run disabled until model B is chosen", async () => {
    await mount();
    expect(runButton().disabled).toBe(true);
    await chooseModelB("AdExNeuron");
    expect(runButton().disabled).toBe(false);
  });

  it("runs model B on the current drive, duration and protocol", async () => {
    await mount();
    await chooseModelB("Izhikevich2007Neuron");
    await act(async () => { runButton().click(); await Promise.resolve(); });
    expect(store.runCompare).toHaveBeenCalledWith({
      model_name: "Izhikevich2007Neuron",
      params: {},
      duration: 500,
      current: 12.5,
      protocol: { kind: "step" },
    });
  });

  it("keeps the run disabled while a simulation is running", async () => {
    store.isSimulating = true;
    await mount();
    await chooseModelB("AdExNeuron");
    expect(runButton().disabled).toBe(true);
  });
});
