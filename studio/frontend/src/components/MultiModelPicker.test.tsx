// @vitest-environment happy-dom
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio multi-model picker offers the whole catalogue

import { act } from "react";
import type React from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

import type { ModelSummary } from "../api/client";
import { useStudioStore } from "../stores/studio";
import MultiModelPicker, { MULTI_MODEL_LIMIT, multiModelChoices } from "./MultiModelPicker";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/**
 * Render into the document with the live store and return the markup.
 *
 * Server rendering reads a zustand store's initial state, not the state a
 * case sets, so these cases mount into a real document instead.
 *
 * @param element - What to render.
 * @returns The rendered markup.
 */
async function mountedMarkup(element: React.ReactElement): Promise<string> {
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  await act(async () => { root.render(element); });
  const html = container.innerHTML;
  await act(async () => { root.unmount(); });
  container.remove();
  return html;
}

const NAMES = Array.from({ length: 185 }, (_, i) => `Model${String(i).padStart(3, "0")}Neuron`);

describe("multiModelChoices", () => {
  it("offers every model, not the first fifty", () => {
    expect(multiModelChoices(NAMES, [], "")).toHaveLength(185);
    expect(multiModelChoices(NAMES, [], "")).toContain("Model184Neuron");
  });

  it("filters by name ignoring case and keeps the chosen models on top", () => {
    expect(multiModelChoices(["AdExNeuron", "LIFNeuron", "ExpLIF"], ["LIFNeuron"], "adex")).toEqual([
      "LIFNeuron",
      "AdExNeuron",
    ]);
  });

  it("drops a chosen name the catalogue no longer lists", () => {
    expect(multiModelChoices(["AdExNeuron"], ["GoneNeuron"], "")).toEqual(["AdExNeuron"]);
  });
});

describe("MultiModelPicker", () => {
  const initial = useStudioStore.getState();
  afterEach(() => {
    useStudioStore.setState(initial, true);
  });

  it("titles itself as the view it fills and lists every catalogue model", async () => {
    useStudioStore.setState({
      models: NAMES.map((name) => ({ name }) as ModelSummary),
      selectedModelName: "Model000Neuron",
    });

    const html = await mountedMarkup(<MultiModelPicker />);

    expect(html).toContain(`Multi-model (0 of ${String(MULTI_MODEL_LIMIT)})`);
    expect(html.match(/type="checkbox"/g)).toHaveLength(185);
    expect(html).toContain(">Model184Neuron<");
    expect(html).toContain("Filter models to overlay");
    expect(html).toContain(">Overlay current<");
  });
});
