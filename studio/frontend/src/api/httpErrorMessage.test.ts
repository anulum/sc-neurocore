// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — tests for the sentences read out of server error details

import { describe, expect, it } from "vitest";

import { errorMessage } from "./http";

describe("server error sentences", () => {
  it("says a numerical model failure in full instead of its code", () => {
    expect(errorMessage({
      error: "model_simulation_failed",
      model: "ATypeKNeuron",
      backend: "python",
      step: 170,
      time_ms: 85,
      diagnostic: "ValueError: membrane candidate left physiological safety bounds",
    }, 422)).toBe(
      "ATypeKNeuron (python) simulation failed at step 170 (85.0 ms): "
        + "ValueError: membrane candidate left physiological safety bounds.",
    );
  });

  it("keeps what it can when fields are missing", () => {
    expect(errorMessage({ error: "model_simulation_failed" }, 422)).toBe("The model simulation failed.");
  });

  it("prefers a reason or message and falls back to the status", () => {
    expect(errorMessage({ reason: "unsupported protocol" }, 422)).toBe("unsupported protocol");
    expect(errorMessage("plain detail", 400)).toBe("plain detail");
    expect(errorMessage(null, 503)).toBe("503");
  });
});
