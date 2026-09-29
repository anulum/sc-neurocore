// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Conversion run result display tests
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import type { TrainingConversionResult as Result } from "../trainingConversion";
import TrainingConversionResult from "./TrainingConversionResult";

vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);

/**
 * Render one result and return the text a reader sees.
 *
 * @param result - The result to show.
 * @returns The rendered text, or an empty string.
 */
async function rendered(result: Result | null, target?: string): Promise<string> {
  const host = document.createElement("div");
  const root = createRoot(host);
  await act(async () => { root.render(<TrainingConversionResult result={result} target={target} />); });
  const text = host.textContent;
  await act(async () => { root.unmount(); });
  return text;
}

describe("conversion result display", () => {
  it("states the converted accuracy first, then the source and the drop", async () => {
    expect(await rendered({ val_accuracy: 0.8125, source_val_accuracy: 0.84375, conversion_accuracy_drop: 0.0312 }))
      .toBe("Converted network: 81.3% validation accuracy (source ANN 84.4%, drop 3.1%). "
        + "Measured on the validation split and sealed in training/conversion_report.json.");
  });

  it("adds the accuracy with coefficients rounded for the run's target", async () => {
    const result = { val_accuracy: 0.8, source_val_accuracy: 0.8, conversion_accuracy_drop: 0, target_accuracy: 0.75 };
    expect(await rendered(result, "ecp5")).toContain(
      "With coefficients rounded for ecp5: 75.0%, sealed in training/target_report.json.",
    );
    expect(await rendered(result)).toContain("rounded for the target: 75.0%");
  });

  it("shows nothing for a spiking run", async () => {
    expect(await rendered(null)).toBe("");
  });
});
