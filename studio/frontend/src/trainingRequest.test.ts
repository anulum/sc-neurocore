// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Training request body tests

import { describe, expect, it } from "vitest";
import type { StudioProjectTrainingConfig } from "./studioProjectState";
import {
  QCFS_STEP_LIMIT,
  conversionProblem,
  readTrainingModelKind,
  readTrainingTargetProfile,
  readTrainingTargetProfiles,
  trainingRequestBody,
} from "./trainingRequest";

const SPIKING: StudioProjectTrainingConfig = {
  dataset: "synthetic", epochs: 1, batch_size: 32, lr: 0.01, hidden: [16], timesteps: 4,
  surrogate: "fast_sigmoid", learn_beta: true, learn_threshold: false,
};

describe("training request body", () => {
  it("sends a spiking run exactly as the form holds it, with no kind", () => {
    expect(trainingRequestBody({ ...SPIKING, model_kind: undefined })).toEqual(SPIKING);
    expect(trainingRequestBody(SPIKING)).not.toHaveProperty("model_kind");
  });

  it("sends a conversion run with its kind and without cell settings", () => {
    const body = trainingRequestBody({ ...SPIKING, model_kind: "qcfs_conversion" });
    expect(body).toEqual({
      model_kind: "qcfs_conversion", dataset: "synthetic", epochs: 1, batch_size: 32, lr: 0.01,
      hidden: [16], timesteps: 4,
    });
  });

  it("sends a target only with a conversion run", () => {
    expect(trainingRequestBody({ ...SPIKING, model_kind: "qcfs_conversion", target_profile: "loihi2" })
      .target_profile).toBe("loihi2");
    expect(trainingRequestBody({ ...SPIKING, target_profile: "loihi2" })).not.toHaveProperty("target_profile");
  });

  it("reads target names and listed profiles, dropping what it cannot use", () => {
    expect(readTrainingTargetProfile("ecp5")).toBe("ecp5");
    expect(readTrainingTargetProfile("")).toBeUndefined();
    expect(readTrainingTargetProfile(5)).toBeUndefined();
    const loihi = { name: "loihi2", q_format: "Q11.12", vendor: "Intel", family: "Loihi", platform_class: "neuromorphic",
      data_width: 24, fraction: 12, signed: true };
    expect(readTrainingTargetProfiles([loihi, null, 3, { name: "", q_format: "Q1.1" }, { name: "x" }])).toEqual([loihi]);
    expect(readTrainingTargetProfiles({ profiles: [] })).toEqual([]);
  });

  it("copies the hidden widths so the body cannot alias the form", () => {
    const body = trainingRequestBody(SPIKING);
    expect(body.hidden).toEqual(SPIKING.hidden);
    expect(body.hidden).not.toBe(SPIKING.hidden);
  });

  it("reads only a kind other than the default", () => {
    expect(readTrainingModelKind("qcfs_conversion")).toBe("qcfs_conversion");
    expect(readTrainingModelKind("spiking")).toBeUndefined();
    expect(readTrainingModelKind(7)).toBeUndefined();
  });

  it("names what the conversion route cannot run", () => {
    expect(conversionProblem(SPIKING)).toBeNull();
    expect(conversionProblem({ ...SPIKING, dataset: "shd" })).toBeNull();
    const conversion = { ...SPIKING, model_kind: "qcfs_conversion" as const };
    expect(conversionProblem(conversion)).toBeNull();
    expect(conversionProblem({ ...conversion, dataset: "mnist" })).toBeNull();
    expect(conversionProblem({ ...conversion, dataset: "nmnist" })).toBe(
      "A conversion run trains on the synthetic or MNIST dataset.",
    );
    expect(conversionProblem({ ...conversion, timesteps: QCFS_STEP_LIMIT })).toBeNull();
    expect(conversionProblem({ ...conversion, timesteps: QCFS_STEP_LIMIT + 1 })).toContain("at most");
  });
});
