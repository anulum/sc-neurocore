// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — short model labels stay distinct

import { describe, expect, it } from "vitest";

import { shortModelLabels } from "./modelShortName";

describe("shortModelLabels", () => {
  it("drops a trailing Neuron or Model suffix", () => {
    const labels = shortModelLabels(["AdExNeuron", "WilsonCowanModel", "WongWangUnit"]);
    expect([...labels.values()]).toEqual(["AdEx", "WilsonCowan", "WongWangUnit"]);
  });

  it("keeps the full names of models that would read alike", () => {
    const labels = shortModelLabels(["AstrocyteModel", "AstrocyteNeuron", "AstrocyteLIFNeuron"]);
    expect(labels.get("AstrocyteModel")).toBe("AstrocyteModel");
    expect(labels.get("AstrocyteNeuron")).toBe("AstrocyteNeuron");
    expect(labels.get("AstrocyteLIFNeuron")).toBe("AstrocyteLIF");
  });

  it("only removes the suffix, never a word inside the name", () => {
    expect(shortModelLabels(["NeuronModelBridge"]).get("NeuronModelBridge")).toBe("NeuronModelBridge");
    expect(shortModelLabels(["ModelNeuron"]).get("ModelNeuron")).toBe("Model");
  });

  it("never shows an empty label", () => {
    expect(shortModelLabels(["Neuron"]).get("Neuron")).toBe("Neuron");
  });
});
