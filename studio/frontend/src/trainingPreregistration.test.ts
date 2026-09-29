// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Preregistered criterion and verdict reading tests

import { describe, expect, it } from "vitest";
import {
  PREREGISTRATION_RATIONALE_MAX,
  preregistrationProblem,
  readTrainingPreregistration,
  readTrainingPreregistrationVerdict,
} from "./trainingPreregistration";

const VERDICT = {
  schema_version: "studio.training-preregistration.v1",
  metric: "val_accuracy",
  direction: "at_least",
  threshold: 0.5,
  observed: 0.71875,
  passed: true,
  preregistration_sha256: "a".repeat(64),
};

describe("preregistered criterion", () => {
  it("names the problem the server would refuse, and nothing otherwise", () => {
    expect(preregistrationProblem({ metric: "val_accuracy", threshold: 1, rationale: "" })).toBeNull();
    expect(preregistrationProblem({ metric: "val_loss", threshold: 12.5, rationale: "" })).toBeNull();
    expect(preregistrationProblem({ metric: "val_accuracy", threshold: 1.5, rationale: "" }))
      .toBe("An accuracy threshold lies between 0 and 1.");
    expect(preregistrationProblem({ metric: "val_loss", threshold: -1, rationale: "" }))
      .toBe("A loss threshold is a finite number at or above 0.");
    expect(preregistrationProblem({ metric: "val_loss", threshold: Number.POSITIVE_INFINITY, rationale: "" }))
      .toBe("A loss threshold is a finite number at or above 0.");
    expect(preregistrationProblem({ metric: "train_accuracy" as "val_loss", threshold: 0, rationale: "" }))
      .toBe("Choose validation accuracy or validation loss.");
    expect(preregistrationProblem({
      metric: "val_loss", threshold: 1, rationale: "x".repeat(PREREGISTRATION_RATIONALE_MAX + 1),
    })).toBe("State the rationale in at most 500 characters.");
  });

  it("judges an accuracy drop only on a conversion run, as a fraction", () => {
    const drop = { metric: "conversion_accuracy_drop" as const, threshold: 0.05, rationale: "" };
    expect(preregistrationProblem(drop, "qcfs_conversion")).toBeNull();
    expect(preregistrationProblem(drop)).toBe("The accuracy-drop criterion judges a conversion run only.");
    expect(preregistrationProblem(drop, "spiking")).toBe("The accuracy-drop criterion judges a conversion run only.");
    expect(preregistrationProblem({ ...drop, threshold: 1.2 }, "qcfs_conversion"))
      .toBe("An accuracy-drop threshold lies between 0 and 1.");
    expect(readTrainingPreregistration(drop)).toEqual(drop);
    expect(readTrainingPreregistrationVerdict({
      ...VERDICT, metric: "conversion_accuracy_drop", direction: "at_most",
    })?.metric).toBe("conversion_accuracy_drop");
    expect(readTrainingPreregistrationVerdict({ ...VERDICT, metric: "conversion_accuracy_drop" })).toBeNull();
  });

  it("reads a stored criterion, dropping the server's own fields, and ignores what it cannot use", () => {
    expect(readTrainingPreregistration({
      metric: "val_loss", threshold: 0.8, rationale: "why", schema_version: "x", sha256: "y",
    })).toEqual({ metric: "val_loss", threshold: 0.8, rationale: "why" });
    expect(readTrainingPreregistration({ metric: "val_accuracy", threshold: 0.5 }))
      .toEqual({ metric: "val_accuracy", threshold: 0.5, rationale: "" });
    for (const value of [undefined, null, [], "val_loss", { metric: "val_loss" },
      { metric: "loss", threshold: 1 }, { metric: "val_accuracy", threshold: 3 }]) {
      expect(readTrainingPreregistration(value)).toBeUndefined();
    }
  });

  it("trusts only a complete verdict whose direction matches its metric", () => {
    expect(readTrainingPreregistrationVerdict(VERDICT)).toEqual(VERDICT);
    expect(readTrainingPreregistrationVerdict({ ...VERDICT, observed: null, passed: false }))
      .toEqual({ ...VERDICT, observed: null, passed: false });
    for (const broken of [
      null, [], { ...VERDICT, schema_version: "v0" }, { ...VERDICT, metric: "loss" },
      { ...VERDICT, direction: "at_most" }, { ...VERDICT, threshold: Number.NaN },
      { ...VERDICT, observed: "0.7" }, { ...VERDICT, passed: "yes" },
      { ...VERDICT, preregistration_sha256: "short" },
    ]) {
      expect(readTrainingPreregistrationVerdict(broken)).toBeNull();
    }
  });
});
