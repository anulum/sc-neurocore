// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Authored compiler controls through DOM and real store actions
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it } from "vitest";
import { useStudioStore } from "../stores/studio";
import { defaultEquationIrConfiguration } from "../equationIrConfig";
import EquationIrOptions from "./EquationIrOptions";

const initial = useStudioStore.getState();
let root: Root;
let container: HTMLDivElement;
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
beforeEach(() => {
  container = document.createElement("div"); document.body.append(container);
  act(() => {
    useStudioStore.setState({ sourceMode: "ode", equationIrConfig: defaultEquationIrConfiguration() });
    root = createRoot(container); root.render(<EquationIrOptions />);
  });
});
afterEach(() => {
  act(() => { root.unmount(); useStudioStore.setState(initial, true); }); container.remove();
});

/**
 * Deliver a native DOM text edit, leaving React's change tracker to observe it.
 *
 * @param control - Rendered input or textarea.
 * @param value - Exact text entered by the user.
 */
function editText(control: HTMLInputElement | HTMLTextAreaElement, value: string): void {
  const prototype = control instanceof HTMLTextAreaElement
    ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  const descriptor = Object.getOwnPropertyDescriptor(prototype, "value");
  if (descriptor?.set === undefined) throw new Error("DOM text control has no native value setter");
  descriptor.set.call(control, value);
  control.dispatchEvent(new Event("input", { bubbles: true }));
}

it("edits exact random keys and partial arithmetic drafts without coercion", () => {
  const seed = container.querySelector<HTMLInputElement>('[aria-label="Compiler random key"]');
  const width = container.querySelector<HTMLInputElement>('[aria-label="Total scalar bits"]');
  if (seed === null || width === null) throw new Error("Compiler inputs absent");
  act(() => { editText(seed, "18446744073709551615"); });
  act(() => { editText(width, ""); });
  expect(useStudioStore.getState().equationIrConfig.model.rng_seed).toBe("18446744073709551615");
  expect(useStudioStore.getState().equationIrConfig.profile.data_width).toBe("");
});

it("edits integration and ordered unit annotations and invalidates old artifacts", () => {
  const integrator = container.querySelector<HTMLSelectElement>('[aria-label="Integrator"]');
  const units = container.querySelector<HTMLTextAreaElement>('[aria-label="State unit declarations"]');
  if (integrator === null || units === null) throw new Error("Scientific inputs absent");
  act(() => { useStudioStore.setState({ irText: "old IR", svSource: "old RTL" }); });
  act(() => { integrator.value = "rk4"; integrator.dispatchEvent(new Event("change", { bubbles: true })); });
  act(() => { editText(units, '[{"name":"v","unit":"millivolt"}]'); });
  const state = useStudioStore.getState();
  expect(state.equationIrConfig.model.method).toBe("rk4");
  expect(state.equationIrConfig.model.state_units).toContain("millivolt");
  expect(state.irText).toBe(""); expect(state.svSource).toBe("");
});

it("explains the equation source requirement in catalogue mode", () => {
  act(() => { useStudioStore.setState({ sourceMode: "model" }); });
  expect(container.textContent).toContain("Choose ODE mode");
  expect(container.querySelector('[aria-label="Integrator"]')).toBeNull();
});

it.each([
  ["Overflow", "overflow", "wrap"],
  ["Product rounding", "rounding", "bankers"],
  ["Function lowering", "function_lowering", "v2"],
  ["Random precision", "random_precision", "exact"],
] as const)("persists the %s arithmetic choice without replacing other drafts", (label, key, value) => {
  const control = container.querySelector<HTMLSelectElement>(`[aria-label="${label}"]`);
  if (control === null) throw new Error(`Missing ${label} control`);
  act(() => { control.value = value; control.dispatchEvent(new Event("change", { bubbles: true })); });
  expect(useStudioStore.getState().equationIrConfig.profile[key]).toBe(value);
  expect(useStudioStore.getState().equationIrConfig.model.rng_seed).toBe("44257");
});

it.each([
  ["Event detection", "detection", "crossing"],
  ["Threshold random generator", "threshold_rng", "philox"],
  ["Units mode", "units", "strict"],
] as const)("persists the %s constructor choice", (label, key, value) => {
  const control = container.querySelector<HTMLSelectElement>(`[aria-label="${label}"]`);
  if (control === null) throw new Error(`Missing ${label} control`);
  act(() => { control.value = value; control.dispatchEvent(new Event("change", { bubbles: true })); });
  expect(useStudioStore.getState().equationIrConfig.model[key]).toBe(value);
});

it("preserves unsigned intent and clears all advanced drafts only on explicit reset", () => {
  const signed = container.querySelector<HTMLInputElement>('input[type="checkbox"]');
  const magnitude = container.querySelector<HTMLInputElement>('[aria-label="Drive unit declaration magnitude"]');
  const reset = container.querySelector<HTMLButtonElement>('button');
  if (signed === null || magnitude === null || reset === null) throw new Error("Compiler controls absent");
  act(() => { signed.click(); });
  act(() => { editText(magnitude, "2.5"); });
  expect(useStudioStore.getState().equationIrConfig.profile.signed).toBe(false);
  expect(useStudioStore.getState().equationIrConfig.model.input_magnitude).toBe("2.5");
  act(() => { reset.click(); });
  expect(useStudioStore.getState().equationIrConfig).toEqual(defaultEquationIrConfiguration());
});
