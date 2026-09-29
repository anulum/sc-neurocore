// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Training configuration interaction tests
// @vitest-environment happy-dom

import { act, useState } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useStudioStore } from "../stores/studio";
import eventData from "../studioEventTrainingData.fixture.json";
import TrainingConfiguration from "./TrainingConfiguration";

vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);

const original = useStudioStore.getState();
afterEach(() => { useStudioStore.setState(original, true); });

/** Render real store-backed settings with a submission readiness consumer. */
function Harness() {
  const state = useStudioStore();
  const [ready, setReady] = useState(false);
  return <><TrainingConfiguration config={state.trainingConfig} surrogates={[]}
    targetProfiles={[{ name: "loihi2", q_format: "Q11.12" }]}
    setConfig={state.setTrainingConfig} onReadyChange={setReady} />
    <button disabled={!ready}>Submit training</button></>;
}

/**
 * Find a labelled control in the rendered public configuration form.
 *
 * @param host - Rendered form root.
 * @param name - Visible label text.
 * @returns The corresponding editable control.
 * @throws {Error} If the control is missing.
 */
function control(host: HTMLElement, name: string): HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement {
  for (const label of host.querySelectorAll("label")) {
    if (label.firstChild?.textContent?.trim() === name) {
      const input = label.querySelector("input,select,textarea");
      if (input instanceof HTMLInputElement || input instanceof HTMLSelectElement || input instanceof HTMLTextAreaElement) return input;
    }
  }
  throw new Error(`Control ${name} missing`);
}

/**
 * Enter a value through the browser event surface, not the component's callbacks.
 *
 * @param element - Rendered editable control.
 * @param value - New value.
 */
async function enter(element: HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement, value: string) {
  const prototype = element instanceof HTMLSelectElement ? HTMLSelectElement.prototype
    : element instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  await act(async () => {
    Reflect.set(prototype, "value", value, element);
    element.dispatchEvent(new Event(element instanceof HTMLSelectElement ? "change" : "input", { bubbles: true }));
  });
}

describe("event training configuration", () => {
  it("imports the declared window, rejects drafts, clears inputs and returns to static training", async () => {
    const host = document.createElement("div");
    const root = createRoot(host);
    try {
      await act(async () => { root.render(<Harness />); });
      const submit = host.querySelector<HTMLButtonElement>("button:last-child");
      expect(submit?.disabled).toBe(false);
      await enter(control(host, "Dataset"), "nmnist");
      expect(submit?.disabled).toBe(true);
      const textarea = control(host, "Event input JSON");
      await enter(textarea, JSON.stringify(eventData));
      await act(async () => { host.querySelectorAll("button")[0]?.click(); });
      expect(useStudioStore.getState().trainingConfig.event_data).toEqual(eventData);
      expect(useStudioStore.getState().trainingConfig.timesteps).toBe(4);
      expect(submit?.disabled).toBe(false);
      await act(async () => { host.querySelectorAll("button")[1]?.click(); });
      expect(useStudioStore.getState().trainingConfig.event_data).toBeUndefined();
      expect(submit?.disabled).toBe(true);
      await enter(textarea, JSON.stringify(eventData));
      await act(async () => { host.querySelectorAll("button")[0]?.click(); });
      await enter(textarea, "{bad json");
      expect(submit?.disabled).toBe(true);
      await act(async () => { host.querySelectorAll("button")[0]?.click(); });
      expect(host.querySelector('[role="alert"]')).not.toBeNull();
      expect(useStudioStore.getState().trainingConfig.event_data).toEqual(eventData);
      await enter(control(host, "Dataset"), "synthetic");
      expect(useStudioStore.getState().trainingConfig.event_data).toBeUndefined();
      expect(submit?.disabled).toBe(false);
    } finally {
      await act(async () => { root.unmount(); });
    }
  });

  it("preserves zero replay settings and refuses empty numerical input without invented defaults", async () => {
    const host = document.createElement("div");
    const root = createRoot(host);
    try {
      await act(async () => { root.render(<Harness />); });
      await enter(control(host, "Seed"), "0");
      await enter(control(host, "Gradient norm limit"), "0");
      expect(useStudioStore.getState().trainingConfig).toMatchObject({ seed: 0, max_grad_norm: 0 });
      await enter(control(host, "Batch Size"), "3");
      expect(useStudioStore.getState().trainingConfig.batch_size).toBe(3);
      await enter(control(host, "Epochs"), "");
      expect(useStudioStore.getState().trainingConfig.epochs).toBe(0);
      expect(host.querySelector<HTMLButtonElement>("button:last-child")?.disabled).toBe(true);
    } finally {
      await act(async () => { root.unmount(); });
    }
  });
});

describe("conversion route", () => {
  it("moves off an event dataset, hides cell settings and offers the accuracy-drop criterion", async () => {
    const host = document.createElement("div");
    const root = createRoot(host);
    try {
      await act(async () => { root.render(<Harness />); });
      const submit = host.querySelector<HTMLButtonElement>("button:last-child");
      await enter(control(host, "Dataset"), "shd");
      expect(submit?.disabled).toBe(true);
      await enter(control(host, "Model"), "qcfs_conversion");
      const state = useStudioStore.getState().trainingConfig;
      expect(state.model_kind).toBe("qcfs_conversion");
      expect(state.dataset).toBe("synthetic");
      expect(state.event_data).toBeUndefined();
      expect(submit?.disabled).toBe(false);
      expect([...(control(host, "Dataset") as HTMLSelectElement).options].map((option) => option.value))
        .toEqual(["synthetic", "mnist"]);
      expect(() => control(host, "Surrogate")).toThrow("Control Surrogate missing");
      expect(host.textContent).toContain("converts it to an");
      const declare = [...host.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')]
        .find((box) => box.parentElement?.textContent.includes("Judge this run"));
      await act(async () => { declare?.click(); });
      await enter(control(host, "Metric"), "conversion_accuracy_drop");
      await enter(control(host, "Threshold"), "0.05");
      expect(submit?.disabled).toBe(false);
      await enter(control(host, "Timesteps"), String(2 ** 32));
      expect(submit?.disabled).toBe(true);
      expect(host.textContent).toContain("timestep budget is at most");
      await enter(control(host, "Timesteps"), "8");
      expect([...(control(host, "Target") as HTMLSelectElement).options].map((option) => option.textContent))
        .toEqual(["No target calibration", "loihi2 (Q11.12)"]);
      await enter(control(host, "Target"), "loihi2");
      expect(useStudioStore.getState().trainingConfig.target_profile).toBe("loihi2");
      await enter(control(host, "Target"), "");
      expect(useStudioStore.getState().trainingConfig.target_profile).toBeUndefined();
      await act(async () => { useStudioStore.getState().setTrainingConfig("target_profile", "ecp5"); });
      expect((control(host, "Target") as HTMLSelectElement).value).toBe("ecp5");
      await enter(control(host, "Model"), "spiking");
      expect(() => control(host, "Target")).toThrow("Control Target missing");
      expect(useStudioStore.getState().trainingConfig.model_kind).toBeUndefined();
      expect(control(host, "Surrogate")).toBeDefined();
      expect(submit?.disabled).toBe(true);
      expect(host.querySelector('[role="alert"]')?.textContent)
        .toBe("The accuracy-drop criterion judges a conversion run only.");
      await enter(control(host, "Model"), "qcfs_conversion");
      await enter(control(host, "Dataset"), "mnist");
      expect(useStudioStore.getState().trainingConfig.dataset).toBe("mnist");
      await enter(control(host, "Model"), "qcfs_conversion");
      expect(useStudioStore.getState().trainingConfig.dataset).toBe("mnist");
    } finally {
      await act(async () => { root.unmount(); });
    }
  });
});

describe("preregistered acceptance criterion", () => {
  it("declares, edits, refuses an unjudgeable bound and withdraws the criterion through the form", async () => {
    const host = document.createElement("div");
    const root = createRoot(host);
    try {
      await act(async () => { root.render(<Harness />); });
      const submit = host.querySelector<HTMLButtonElement>("button:last-child");
      const declare = [...host.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')]
        .find((box) => box.parentElement?.textContent.includes("Judge this run"));
      expect(declare).toBeDefined();
      await act(async () => { declare?.click(); });
      expect(useStudioStore.getState().trainingConfig.preregistration).toEqual({
        metric: "val_accuracy", threshold: 0.5, rationale: "",
      });
      expect(submit?.disabled).toBe(false);
      await enter(control(host, "Threshold"), "2");
      expect(submit?.disabled).toBe(true);
      expect(host.querySelector('[role="alert"]')?.textContent).toBe("An accuracy threshold lies between 0 and 1.");
      await enter(control(host, "Metric"), "val_loss");
      expect(submit?.disabled).toBe(false);
      expect(host.querySelector('[role="alert"]')).toBeNull();
      await enter(control(host, "Threshold"), "");
      expect(submit?.disabled).toBe(true);
      expect(host.querySelector('[role="alert"]')?.textContent).toBe("A loss threshold is a finite number at or above 0.");
      await enter(control(host, "Threshold"), "0.8");
      await enter(control(host, "Rationale"), "loss below 0.8 after one epoch");
      expect(useStudioStore.getState().trainingConfig.preregistration).toEqual({
        metric: "val_loss", threshold: 0.8, rationale: "loss below 0.8 after one epoch",
      });
      expect(submit?.disabled).toBe(false);
      await act(async () => { declare?.click(); });
      expect(useStudioStore.getState().trainingConfig.preregistration).toBeUndefined();
      expect(control(host, "Dataset")).toBeDefined();
      expect(() => control(host, "Metric")).toThrow("Control Metric missing");
    } finally {
      await act(async () => { root.unmount(); });
    }
  });
});
