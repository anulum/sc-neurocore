// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — the preregistered acceptance criterion of a training run
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { TrainingModelKind, TrainingPreregistration } from "../api/client";
import { PREREGISTRATION_RATIONALE_MAX, preregistrationProblem } from "../trainingPreregistration";
import TrainingPreregistrationInput from "./TrainingPreregistrationInput";

let root: Root | undefined;
let container: HTMLDivElement;
const onChange = vi.fn();

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  onChange.mockClear();
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
 * Render the criterion controls.
 *
 * @param value - The criterion, or none.
 * @param modelKind - The run's kind.
 */
async function render(value: TrainingPreregistration | undefined, modelKind?: TrainingModelKind): Promise<void> {
  await act(async () => {
    root?.render(<TrainingPreregistrationInput value={value} onChange={onChange} modelKind={modelKind} />);
    await Promise.resolve();
  });
}

/**
 * Set a control's value through the native setter React listens for.
 *
 * @param element - The control.
 * @param text - Its new value.
 * @param event - The event React reads for that control.
 */
async function set(element: HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement, text: string, event = "input"): Promise<void> {
  const prototype = element instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype
    : element instanceof HTMLSelectElement ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
  await act(async () => {
    Object.getOwnPropertyDescriptor(prototype, "value")?.set?.call(element, text);
    element.dispatchEvent(new Event(event, { bubbles: true }));
    await Promise.resolve();
  });
}

const CRITERION: TrainingPreregistration = { metric: "val_accuracy", threshold: 0.8, rationale: "Beats chance by a margin." };

describe("TrainingPreregistrationInput", () => {
  it("declares nothing by default and offers to declare a criterion", async () => {
    await render(undefined);
    expect(container.querySelector("legend")?.textContent).toBe("Preregistered acceptance criterion");
    const box = container.querySelector<HTMLInputElement>("input[type=checkbox]");
    expect(box?.checked).toBe(false);
    expect(container.querySelector("select")).toBeNull();
    await act(async () => { box?.click(); await Promise.resolve(); });
    expect(onChange).toHaveBeenCalledWith({ metric: "val_accuracy", threshold: 0.5, rationale: "" });
  });

  it("withdraws the criterion when the box is cleared", async () => {
    await render(CRITERION);
    await act(async () => { container.querySelector<HTMLInputElement>("input[type=checkbox]")?.click(); await Promise.resolve(); });
    expect(onChange).toHaveBeenCalledWith(undefined);
  });

  it("offers the accuracy-drop criterion only for a conversion run", async () => {
    await render(CRITERION);
    const values = () => [...(container.querySelector("select")?.options ?? [])].map((option) => option.value);
    expect(values()).toEqual(["val_accuracy", "val_loss"]);
    await render(CRITERION, "qcfs_conversion");
    expect(values()).toEqual(["val_accuracy", "val_loss", "conversion_accuracy_drop"]);
  });

  it("carries each edit into a whole new criterion", async () => {
    await render(CRITERION);
    const select = container.querySelector<HTMLSelectElement>('select[aria-label="Criterion metric"]');
    const threshold = container.querySelector<HTMLInputElement>('input[aria-label="Criterion threshold"]');
    const rationale = container.querySelector<HTMLTextAreaElement>('textarea[aria-label="Criterion rationale"]');
    if (select === null || threshold === null || rationale === null) throw new Error("the criterion controls are missing");
    await set(select, "val_loss", "change");
    expect(onChange).toHaveBeenLastCalledWith({ ...CRITERION, metric: "val_loss" });
    await set(threshold, "0.25");
    expect(onChange).toHaveBeenLastCalledWith({ ...CRITERION, threshold: 0.25 });
    await set(rationale, "Stated in advance.");
    expect(onChange).toHaveBeenLastCalledWith({ ...CRITERION, rationale: "Stated in advance." });
    expect(rationale.maxLength).toBe(PREREGISTRATION_RATIONALE_MAX);
  });

  it("turns an emptied threshold into a value the criterion check refuses, and says why", async () => {
    await render(CRITERION);
    const threshold = container.querySelector<HTMLInputElement>('input[aria-label="Criterion threshold"]');
    if (threshold === null) throw new Error("no threshold");
    await set(threshold, "");
    const sent = onChange.mock.lastCall?.[0] as TrainingPreregistration;
    expect(Number.isNaN(sent.threshold)).toBe(true);
    const problem = preregistrationProblem(sent);
    expect(problem).not.toBeNull();
    await render(sent);
    expect(container.querySelector('[role="alert"]')?.textContent).toBe(problem);
  });

  it("raises no alert for a criterion that can be sent", async () => {
    await render(CRITERION);
    expect(preregistrationProblem(CRITERION)).toBeNull();
    expect(container.querySelector('[role="alert"]')).toBeNull();
  });
});
