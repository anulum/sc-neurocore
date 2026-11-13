// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — the model's silicon capability strip, mounted with a stand-in for the capability request
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ModelCapabilities, ModelCapabilityOperation } from "../api/client";
import ModelCapabilitiesStrip from "./ModelCapabilities";

const requests = vi.hoisted(() => ({
  pending: new Map<string, { resolve: (value: unknown) => void; reject: (reason: unknown) => void }>(),
}));

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    fetchModelCapabilities: (name: string) => new Promise((resolve, reject) => {
      requests.pending.set(name, { resolve, reject });
    }),
  };
});

const disabled = (reason: string): ModelCapabilityOperation => ({ enabled: false, reason });

/**
 * A capability matrix with compilation enabled and the other operations disabled for a stated reason.
 *
 * @param model - The model the matrix is for.
 * @returns The matrix.
 */
function matrix(model: string): ModelCapabilities {
  return {
    schema_version: "sc-neurocore.studio.model-capabilities.v1",
    model,
    operations: {
      compile: { enabled: true, integrators: ["rk4"], q_formats: ["Q16.16"] } as ModelCapabilityOperation,
      cosimulate: disabled("no mirrored combination"),
      synthesise: disabled("no synthesis target"),
      place_and_route: disabled("no place-and-route target"),
      formal: disabled("no formal job"),
    },
  };
}

let root: Root | undefined;
let container: HTMLDivElement;

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  requests.pending.clear();
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root?.unmount(); await Promise.resolve(); });
  document.body.replaceChildren();
  vi.unstubAllGlobals();
});

/**
 * Render the strip for a model.
 *
 * @param modelName - The selected model.
 */
async function show(modelName: string): Promise<void> {
  await act(async () => { root?.render(<ModelCapabilitiesStrip modelName={modelName} />); await Promise.resolve(); });
}

/**
 * Settle the pending request for a model.
 *
 * @param name - The model whose request settles.
 * @param outcome - The matrix to return, or an error to fail with.
 */
async function settle(name: string, outcome: ModelCapabilities | Error): Promise<void> {
  const request = requests.pending.get(name);
  if (request === undefined) throw new Error(`no request was made for ${name}`);
  await act(async () => {
    if (outcome instanceof Error) request.reject(outcome);
    else request.resolve(outcome);
    await Promise.resolve();
  });
}

/** The strip's rows as text. */
function rows(): string[] {
  return [...container.querySelectorAll("li")].map((item) => item.textContent);
}

describe("ModelCapabilitiesStrip", () => {
  it("draws nothing until the matrix arrives, then one row per operation in pipeline order", async () => {
    await show("LIFNeuron");
    expect(container.innerHTML).toBe("");
    await settle("LIFNeuron", matrix("LIFNeuron"));
    const list = container.querySelector("ul");
    expect(list?.getAttribute("aria-label")).toBe("Silicon operations for this model");
    expect(rows()).toEqual([
      "✓ compile — rk4 · Q16.16",
      "✗ co-simulate — not available: no mirrored combination",
      "✗ synthesise — not available: no synthesis target",
      "✗ place & route — not available: no place-and-route target",
      "✗ formal — not available: no formal job",
    ]);
  });

  it("keeps the tick and cross out of the accessible text", async () => {
    await show("LIFNeuron");
    await settle("LIFNeuron", matrix("LIFNeuron"));
    for (const mark of container.querySelectorAll("li > span:first-child")) {
      expect(mark.getAttribute("aria-hidden")).toBe("true");
    }
  });

  it("draws nothing when the request fails", async () => {
    await show("LIFNeuron");
    await settle("LIFNeuron", new Error("offline"));
    expect(container.innerHTML).toBe("");
  });

  it("drops a late answer for a model that is no longer selected", async () => {
    await show("LIFNeuron");
    await show("AdExNeuron");
    await settle("LIFNeuron", matrix("LIFNeuron"));
    expect(container.innerHTML).toBe("");
    await settle("AdExNeuron", matrix("AdExNeuron"));
    expect(rows()).toHaveLength(5);
  });

  it("ignores a late failure for a model that is no longer selected", async () => {
    await show("LIFNeuron");
    await show("AdExNeuron");
    await settle("AdExNeuron", matrix("AdExNeuron"));
    await settle("LIFNeuron", new Error("offline"));
    expect(rows()).toHaveLength(5);
  });

  it("clears the previous model's rows while the next model's matrix is on its way", async () => {
    await show("LIFNeuron");
    await settle("LIFNeuron", matrix("LIFNeuron"));
    expect(rows()).toHaveLength(5);
    await show("AdExNeuron");
    expect(container.innerHTML).toBe("");
  });
});
