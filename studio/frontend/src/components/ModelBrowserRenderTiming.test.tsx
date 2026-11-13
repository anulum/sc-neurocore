// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — the catalogue's render timing mark, through the mounted catalogue
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { initialModelScanJobState } from "../modelScanJob";
import ModelBrowser from "./ModelBrowser";

const store = vi.hoisted(() => ({ models: [] as Record<string, unknown>[] }));

vi.mock("../useModelScanJob", () => ({
  useModelScanJob: () => ({
    busy: false,
    canSubmit: true,
    phaseLabel: "Scan",
    startScan: () => undefined,
    state: initialModelScanJobState(),
  }),
}));

vi.mock("../stores/studio", () => ({
  useStudioStore: () => ({
    models: store.models,
    selectedModelName: null,
    modelFilter: "",
    loadModels: () => undefined,
    selectModel: () => undefined,
    setModelFilter: () => undefined,
  }),
}));

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    fetchModelFacets: async () => null,
    // The query answer is not part of these cases; it stays pending.
    queryModels: () => new Promise(() => undefined),
  };
});

const MODEL = {
  name: "LIFNeuron", category: "IF", family: "IF", tier: 2, science_tier: 3, silicon_tier: null,
  science_label: "S3", silicon_label: "none", validation_metric: "parity", integration_method: "euler",
  terminal_silicon_tier: "H1", terminal_reason: "", behavior_tags: ["tonic"], description: "LIF",
  maturity: "validated", evidence_kind: "publication", provenance: null, state_var_names: ["v"], n_params: 2,
};

let root: Root | undefined;
let container: HTMLDivElement;

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  store.models = [];
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root?.unmount(); await Promise.resolve(); });
  document.body.replaceChildren();
  performance.clearMarks();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

/**
 * Render the catalogue with a list of models.
 *
 * @param models - The models the store holds.
 */
async function show(models: Record<string, unknown>[]): Promise<void> {
  store.models = models;
  await act(async () => {
    root?.render(<ModelBrowser />);
    await Promise.resolve();
  });
}

/** The catalogue-drawn marks recorded so far. */
function catalogueMarks(): number {
  return performance.getEntriesByName("studio:catalogue-drawn", "mark").length;
}

describe("ModelBrowser render timing", () => {
  it("marks the catalogue once, when its rows are first drawn", async () => {
    await show([]);
    expect(catalogueMarks()).toBe(0);
    await show([MODEL]);
    expect(catalogueMarks()).toBe(1);
    await show([MODEL, { ...MODEL, name: "ThetaNeuron" }]);
    expect(catalogueMarks()).toBe(1);
  });

  it("draws the same markup whether or not the timeline takes marks", async () => {
    await show([MODEL]);
    const marked = container.innerHTML;
    await act(async () => { root?.unmount(); await Promise.resolve(); });
    root = createRoot(container);
    vi.spyOn(performance, "mark").mockImplementation(() => {
      throw new Error("no timeline");
    });
    await show([MODEL]);
    expect(container.innerHTML).toBe(marked);
  });
});
