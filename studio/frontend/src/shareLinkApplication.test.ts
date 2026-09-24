// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { describe, expect, it } from "vitest";

import { studioModelLinkDecision, studioShareLinkDecision } from "./shareLinkApplication";
import type { StudioStartupHashState } from "./studioUrlState";

const LINK: StudioStartupHashState = {
  selectedModelName: "LapicqueNeuron",
  current: 12.5,
  duration: 250,
  protocol: "step",
};

describe("what a share link asks for", () => {
  it("asks for nothing when the page was opened without one", () => {
    expect(studioShareLinkDecision(null, ["LapicqueNeuron"])).toEqual({ kind: "none" });
  });

  it("carries every setting the link encoded, not only the model", () => {
    // The current, duration and protocol are what make the link reproduce a
    // reading rather than merely open a page.
    expect(studioShareLinkDecision(LINK, ["AdExNeuron", "LapicqueNeuron"])).toEqual({
      kind: "apply",
      modelName: "LapicqueNeuron",
      current: 12.5,
      duration: 250,
      protocol: "step",
    });
  });

  it("names the model when this catalogue does not hold it", () => {
    const decision = studioShareLinkDecision(LINK, ["AdExNeuron"]);

    expect(decision.kind).toBe("unknown-model");
    expect(decision.kind === "unknown-model" && decision.modelName).toBe("LapicqueNeuron");
    expect(decision.kind === "unknown-model" && decision.message).toContain("LapicqueNeuron");
    expect(decision.kind === "unknown-model" && decision.message).toContain("renamed");
  });

  it("treats an empty catalogue as not holding the model, not as no link", () => {
    // The caller waits for the catalogue before asking; if it ever did not,
    // this must still be a refusal by name rather than a silent no-op.
    expect(studioShareLinkDecision(LINK, []).kind).toBe("unknown-model");
  });
});

describe("what a model link asks for", () => {
  it("does nothing without a model link", () => {
    expect(studioModelLinkDecision(null, ["AdExNeuron"])).toEqual({ kind: "none" });
  });

  it("selects a model the catalogue holds and asks for nothing else", () => {
    expect(studioModelLinkDecision("AdExNeuron", ["AdExNeuron"])).toEqual({
      kind: "select",
      modelName: "AdExNeuron",
    });
  });

  it("refuses a name the catalogue does not hold, by name", () => {
    const decision = studioModelLinkDecision("RetiredNeuron", ["AdExNeuron"]);
    expect(decision.kind).toBe("unknown-model");
    expect(decision.kind === "unknown-model" && decision.message).toContain('"RetiredNeuron"');
  });
});
