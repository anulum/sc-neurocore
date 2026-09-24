// @vitest-environment happy-dom
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — The first-run tour states the catalogue it was given

/**
 * The tour names the number of models the catalogue delivered, never a number
 * written into the tour, and says nothing numeric before the catalogue arrives.
 */

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import OnboardingOverlay from "./OnboardingOverlay";

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  localStorage.clear();
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root.unmount(); });
  container.remove();
});

/**
 * Show the tour with `modelCount` models loaded and open its catalogue step.
 *
 * @param modelCount - The models loaded from the catalogue.
 * @returns The catalogue step's text.
 */
async function catalogueStep(modelCount: number): Promise<string> {
  await act(async () => { root.render(<OnboardingOverlay modelCount={modelCount} />); });
  const next = [...container.querySelectorAll("button")].find((button) => button.textContent === "Next");
  expect(next).toBeDefined();
  await act(async () => { next?.click(); });
  expect(container.querySelector("h2")?.textContent).toBe("Model Browser");
  return container.querySelector("p")?.textContent ?? "";
}

describe("OnboardingOverlay", () => {
  it("names the number of models the catalogue delivered", async () => {
    expect(await catalogueStep(185)).toMatch(
      /^Browse the catalogue's 185 neuron models by category\. /,
    );
  });

  it("states no number before the catalogue has arrived, and the number once it has", async () => {
    expect(await catalogueStep(0)).toMatch(/^Browse the catalogue's neuron models by category\. /);

    await act(async () => { root.render(<OnboardingOverlay modelCount={212} />); });
    expect(container.querySelector("p")?.textContent).toMatch(
      /^Browse the catalogue's 212 neuron models by category\. /,
    );
  });

  it("keeps a step without a count as written", async () => {
    await act(async () => { root.render(<OnboardingOverlay modelCount={185} />); });
    expect(container.querySelector("h2")?.textContent).toBe("Welcome to SC-NeuroCore Studio");
    expect(container.querySelector("p")?.textContent).toMatch(/^Design, train, compile/);
  });
});
