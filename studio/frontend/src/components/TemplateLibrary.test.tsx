// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — the ODE template picker, mounted with its store
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import TemplateLibrary from "./TemplateLibrary";

const store = vi.hoisted(() => ({
  templates: [] as { name: string; description: string }[],
  loadTemplates: vi.fn(async () => undefined),
  selectTemplate: vi.fn(),
}));

vi.mock("../stores/studio", () => ({
  useStudioStore: () => store,
}));

let root: Root | undefined;
let container: HTMLDivElement;

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  store.templates = [
    { name: "lif", description: "Leaky integrate-and-fire" },
    { name: "fhn", description: "FitzHugh-Nagumo" },
  ];
  store.loadTemplates.mockClear();
  store.selectTemplate.mockClear();
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root?.unmount(); await Promise.resolve(); });
  document.body.replaceChildren();
  vi.unstubAllGlobals();
});

/** Mount the picker and let its loading effect run. */
async function mount(): Promise<HTMLSelectElement> {
  await act(async () => { root?.render(<TemplateLibrary />); await Promise.resolve(); });
  const select = container.querySelector("select");
  if (select === null) throw new Error("the picker drew no select");
  return select;
}

/**
 * Choose a value as a person would, through the change event React listens to.
 *
 * @param select - The picker.
 * @param value - The option's value.
 */
async function choose(select: HTMLSelectElement, value: string): Promise<void> {
  await act(async () => {
    select.value = value;
    select.dispatchEvent(new Event("change", { bubbles: true }));
    await Promise.resolve();
  });
}

describe("TemplateLibrary", () => {
  it("loads the templates once when it is mounted", async () => {
    await mount();
    expect(store.loadTemplates).toHaveBeenCalledTimes(1);
  });

  it("offers each template by its description, after a placeholder that cannot be chosen", async () => {
    const select = await mount();
    const options = [...select.options];
    expect(options.map((option) => [option.value, option.textContent, option.disabled])).toEqual([
      ["", "ODE templates...", true],
      ["lif", "Leaky integrate-and-fire", false],
      ["fhn", "FitzHugh-Nagumo", false],
    ]);
    expect(select.value).toBe("");
  });

  it("selects the chosen template by its name", async () => {
    const select = await mount();
    await choose(select, "fhn");
    expect(store.selectTemplate).toHaveBeenCalledWith("fhn");
    expect(store.selectTemplate).toHaveBeenCalledTimes(1);
  });

  it("selects nothing when the value goes back to the placeholder", async () => {
    const select = await mount();
    await choose(select, "");
    expect(store.selectTemplate).not.toHaveBeenCalled();
  });

  it("offers only the placeholder while no template has loaded", async () => {
    store.templates = [];
    const select = await mount();
    expect(select.options).toHaveLength(1);
  });
});
