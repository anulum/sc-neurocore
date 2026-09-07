// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Experiment identity contract

import { describe, expect, it } from "vitest";

import { studioExperimentKey, studioResultIsCurrent } from "./studioExperimentKey";
import type { StudioSimulationConfigInput } from "./studioSimulationConfig";

/**
 * Build a configuration, overridden field by field.
 *
 * @param overrides - The fields to change.
 * @returns The configuration.
 */
function config(
  overrides: Partial<StudioSimulationConfigInput> = {},
): StudioSimulationConfigInput {
  return {
    current: 10,
    dt: 0.1,
    duration: 100,
    equations: ["dv/dt = -(v - E_L) / tau_m + I / C"],
    frequencyHz: 10,
    modelParams: { tau_m: 10, v_rest: -65 },
    odeInit: { v: -65 },
    odeParams: { C: 1, E_L: -65, tau_m: 10 },
    protocol: "constant",
    reset: "v = -65",
    seed: null,
    selectedModelName: "SCLapicqueLIFNeuron",
    sourceMode: "model",
    threshold: "v > -50",
    trial: "replay",
    ...overrides,
  };
}

describe("the experiment key", () => {
  it("is the same for two configurations that would submit the same run", () => {
    expect(studioExperimentKey(config())).toBe(studioExperimentKey(config()));
  });

  it("does not depend on the order the fields were written in", () => {
    const written = config();
    const rewritten: StudioSimulationConfigInput = {
      trial: written.trial,
      threshold: written.threshold,
      sourceMode: written.sourceMode,
      selectedModelName: written.selectedModelName,
      seed: written.seed,
      reset: written.reset,
      protocol: written.protocol,
      odeParams: written.odeParams,
      odeInit: written.odeInit,
      modelParams: written.modelParams,
      frequencyHz: written.frequencyHz,
      equations: written.equations,
      duration: written.duration,
      dt: written.dt,
      current: written.current,
    };

    expect(studioExperimentKey(rewritten)).toBe(studioExperimentKey(written));
  });

  it.each([
    ["the model", config({ selectedModelName: "SCIzhikevichNeuron" })],
    ["a model parameter", config({ modelParams: { tau_m: 20, v_rest: -65 } })],
    ["the step", config({ dt: 0.05 })],
    ["the duration", config({ duration: 250 })],
    ["the injected current", config({ current: 12 })],
    ["the protocol", config({ protocol: "sine" })],
    ["the sine frequency", config({ frequencyHz: 40 })],
    ["the seed", config({ seed: 7 })],
    ["the trial policy", config({ trial: "fresh" })],
    ["the source mode", config({ sourceMode: "ode" })],
    ["the equations", config({ equations: ["dv/dt = -v"] })],
    ["the threshold", config({ threshold: "v > -40" })],
    ["the reset", config({ reset: "v = -70" })],
    ["an ODE parameter", config({ odeParams: { C: 2, E_L: -65, tau_m: 10 } })],
    ["an initial value", config({ odeInit: { v: -70 } })],
  ])("changes when %s changes", (_label, changed) => {
    expect(studioExperimentKey(changed)).not.toBe(studioExperimentKey(config()));
  });

  it("changes when a field of the inactive branch changes", () => {
    // A workspace switched from a model to an ODE and back is the same
    // experiment only if it came back to the same fields. Reading the active
    // branch alone would call those two states identical.
    const withOtherOde = config({ odeParams: { C: 1, E_L: -60, tau_m: 10 } });

    expect(studioExperimentKey(withOtherOde)).not.toBe(studioExperimentKey(config()));
  });
});

describe("whether a result is current", () => {
  it("accepts a result recorded under the configuration in force", () => {
    const key = studioExperimentKey(config());

    expect(studioResultIsCurrent(key, key)).toBe(true);
  });

  it("refuses a result recorded under a configuration that has since changed", () => {
    const recorded = studioExperimentKey(config());
    const current = studioExperimentKey(config({ dt: 0.05 }));

    expect(studioResultIsCurrent(recorded, current)).toBe(false);
  });

  it("refuses a result that recorded no key at all", () => {
    // No key means no evidence that the result belongs to what is on screen.
    expect(studioResultIsCurrent(null, studioExperimentKey(config()))).toBe(false);
  });
});
