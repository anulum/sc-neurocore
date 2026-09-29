// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — saturated network populations are named, not reported as rates

import { describe, expect, it } from "vitest";

import { saturatedPopulations, saturationWarning } from "./networkSaturation";

describe("saturatedPopulations", () => {
  it("names the populations spiking on half the steps or more", () => {
    // The old canvas defaults at dt 0.5 ms: 1290 Hz is 64.5 % of the steps.
    const found = saturatedPopulations(
      [{ label: "Exc 0", mean_rate_hz: 1290 }, { label: "Inh 1", mean_rate_hz: 28 }],
      0.5,
    );
    expect(found).toEqual([{ label: "Exc 0", stepFraction: 0.645 }]);
  });

  it("finds nothing in a physiological network or without a step", () => {
    expect(saturatedPopulations([{ label: "E", mean_rate_hz: 30 }], 0.1)).toEqual([]);
    expect(saturatedPopulations([{ label: "E", mean_rate_hz: 5000 }], 0)).toEqual([]);
  });
});

describe("saturationWarning", () => {
  it("says the time step sets the rate and what to change", () => {
    expect(saturationWarning([{ label: "Exc 0", stepFraction: 0.645 }])).toBe(
      "Exc 0 spikes on 65 % of steps: the time step sets this rate, not the model. " +
        "The network has run away; weaken excitation or strengthen inhibition.",
    );
    expect(saturationWarning([])).toBeNull();
  });
});
