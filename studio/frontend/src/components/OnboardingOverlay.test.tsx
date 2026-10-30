// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — The first-run tour states the catalogue it was given
// @vitest-environment happy-dom

/**
 * The tour names the number of models the catalogue delivered, never a number
 * written into the tour, and says nothing numeric before the catalogue arrives.
 */

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import OnboardingOverlay from "./OnboardingOverlay";

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  localStorage.clear();
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root.unmount(); });
  container.remove();
  vi.unstubAllGlobals();
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
  it("navigates progress controls and Back while keeping the current step named", async () => {
    await act(async () => { root.render(<OnboardingOverlay modelCount={185} />); });
    const last = container.querySelector<HTMLButtonElement>('[aria-label="Go to step 8 of 8"]');
    if (last === null) throw new Error("The tour has no final progress control");
    await act(async () => { last.click(); });
    expect(container.querySelector("h2")?.textContent).toBe("Keyboard Shortcuts");
    expect(last.getAttribute("aria-current")).toBe("step");
    const back = [...container.querySelectorAll("button")].find((button) => button.textContent === "Back");
    if (back === undefined) throw new Error("The tour has no Back action");
    await act(async () => { back.click(); });
    expect(container.querySelector("h2")?.textContent).toBe("Training Monitor");
    const first = container.querySelector<HTMLButtonElement>('[aria-label="Go to step 1 of 8"]');
    if (first === null) throw new Error("The tour has no first progress control");
    await act(async () => { first.click(); });
    expect(container.querySelector("h2")?.textContent).toBe("Welcome to SC-NeuroCore Studio");
    expect(back.disabled).toBe(true);
  });

  it("opens a named native modal and keeps its keys out of the workbench", async () => {
    const keys: string[] = [];
    const observe = (event: KeyboardEvent) => { keys.push(event.key); };
    window.addEventListener("keydown", observe);
    try {
      await act(async () => { root.render(<OnboardingOverlay modelCount={185} />); });
      const dialog = container.querySelector("dialog");
      if (dialog === null) throw new Error("The first-run tour did not open");
      expect(dialog.open).toBe(true);
      expect(document.getElementById(dialog.getAttribute("aria-labelledby") ?? "")?.textContent)
        .toBe("Welcome to SC-NeuroCore Studio");
      expect(document.getElementById(dialog.getAttribute("aria-describedby") ?? "")?.textContent)
        .toMatch(/^Design, train, compile/);
      await act(async () => {
        dialog.dispatchEvent(new KeyboardEvent("keydown", { key: " ", bubbles: true }));
        dialog.dispatchEvent(new KeyboardEvent("keydown", { key: "3", bubbles: true }));
      });
      expect(keys).toEqual([]);
      await act(async () => { dialog.close(); });
      expect(container.querySelector("dialog")).toBeNull();
      expect(localStorage.getItem("sc-studio-onboarding-dismissed")).toBe("true");
    } finally {
      window.removeEventListener("keydown", observe);
    }
  });

  it("wraps both Tab boundaries without intercepting ordinary internal navigation", async () => {
    await act(async () => { root.render(<OnboardingOverlay modelCount={185} />); });
    const buttons = [...container.querySelectorAll<HTMLButtonElement>("button:not(:disabled)")];
    const first = buttons[0];
    const last = buttons[buttons.length - 1];
    const middle = buttons[1];
    if (first === undefined || last === undefined || middle === undefined) throw new Error("The tour has no keyboard route");
    first.focus();
    const backwards = new KeyboardEvent("keydown", { key: "Tab", shiftKey: true, bubbles: true, cancelable: true });
    first.dispatchEvent(backwards);
    expect(backwards.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(last);
    const forwards = new KeyboardEvent("keydown", { key: "Tab", bubbles: true, cancelable: true });
    last.dispatchEvent(forwards);
    expect(forwards.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(first);
    middle.focus();
    const internal = new KeyboardEvent("keydown", { key: "Tab", bubbles: true, cancelable: true });
    middle.dispatchEvent(internal);
    expect(internal.defaultPrevented).toBe(false);
    expect(document.activeElement).toBe(middle);
  });

  it("keeps a dismissed tour closed after a fresh mount", async () => {
    localStorage.setItem("sc-studio-onboarding-dismissed", "true");
    await act(async () => { root.render(<OnboardingOverlay modelCount={185} />); });
    expect(container.querySelector("dialog")).toBeNull();
  });

  it("remembers Skip tour and completing the last step", async () => {
    await act(async () => { root.render(<OnboardingOverlay modelCount={185} />); });
    const skip = [...container.querySelectorAll("button")].find((button) => button.textContent === "Skip tour");
    if (skip === undefined) throw new Error("The tour has no Skip tour action");
    await act(async () => { skip.click(); });
    expect(container.querySelector("dialog")).toBeNull();
    expect(localStorage.getItem("sc-studio-onboarding-dismissed")).toBe("true");
    localStorage.removeItem("sc-studio-onboarding-dismissed");
    await act(async () => { root.render(<OnboardingOverlay key="new-tour" modelCount={185} />); });
    for (let index = 0; index < 8; index += 1) {
      const next = [...container.querySelectorAll("button")].find((button) =>
        button.textContent === (index === 7 ? "Get Started" : "Next"));
      if (next === undefined) throw new Error("The tour has no next step action");
      await act(async () => { next.click(); });
    }
    expect(container.querySelector("dialog")).toBeNull();
    expect(localStorage.getItem("sc-studio-onboarding-dismissed")).toBe("true");
  });

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
