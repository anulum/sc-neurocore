// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Conversion run result reading tests

import { describe, expect, it } from "vitest";
import { readTrainingConversionResult } from "./trainingConversion";

const FINAL = {
  train_loss: 0.4, train_accuracy: 0.9, val_loss: 0.5,
  val_accuracy: 0.8125, source_val_accuracy: 0.84375, conversion_accuracy_drop: 0.0312,
};

describe("conversion result", () => {
  it("reads the converted and source accuracy together", () => {
    expect(readTrainingConversionResult(FINAL)).toEqual({
      val_accuracy: 0.8125, source_val_accuracy: 0.84375, conversion_accuracy_drop: 0.0312,
    });
    expect(readTrainingConversionResult({ ...FINAL, conversion_accuracy_drop: -0.05 })
      ?.conversion_accuracy_drop).toBe(-0.05);
  });

  it("reads a target accuracy only when it is a fraction", () => {
    expect(readTrainingConversionResult({ ...FINAL, target_accuracy: 0.75 })?.target_accuracy).toBe(0.75);
    expect(readTrainingConversionResult({ ...FINAL, target_accuracy: 3 })).not.toHaveProperty("target_accuracy");
    expect(readTrainingConversionResult(FINAL)).not.toHaveProperty("target_accuracy");
  });

  it("shows nothing for a spiking run or metrics it cannot trust", () => {
    const spiking = { train_loss: 0.4, train_accuracy: 0.9, val_loss: 0.5, val_accuracy: 0.8125 };
    for (const value of [
      null, undefined, [], "0.8", spiking,
      { ...FINAL, val_accuracy: 1.5 },
      { ...FINAL, source_val_accuracy: Number.NaN },
      { ...FINAL, conversion_accuracy_drop: -1.5 },
      { ...FINAL, conversion_accuracy_drop: "0" },
    ]) {
      expect(readTrainingConversionResult(value)).toBeNull();
    }
  });
});
