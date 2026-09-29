// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Preregistered verdict display tests
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import type { TrainingPreregistrationVerdict as Verdict } from "../api/client";
import TrainingPreregistrationVerdict from "./TrainingPreregistrationVerdict";

vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);

const MET: Verdict = {
  schema_version: "studio.training-preregistration.v1",
  metric: "val_accuracy", direction: "at_least", threshold: 0.5,
  observed: 0.71875, passed: true, preregistration_sha256: "0123456789ab".padEnd(64, "c"),
};

/**
 * Render one verdict and return the text a reader sees.
 *
 * @param verdict - The verdict to show.
 * @returns The rendered text, or an empty string.
 */
async function rendered(verdict: Verdict | null): Promise<string> {
  const host = document.createElement("div");
  const root = createRoot(host);
  await act(async () => { root.render(<TrainingPreregistrationVerdict verdict={verdict} />); });
  const text = host.textContent;
  await act(async () => { root.unmount(); });
  return text;
}

describe("preregistered verdict", () => {
  it("states a met criterion with its direction, observation and stored digest", async () => {
    expect(await rendered(MET)).toBe(
      "Criterion met: validation accuracy ≥ 0.5, observed 0.71875. Criterion stored before the run as 0123456789ab.",
    );
  });

  it("states a missed loss criterion and a non-finite observation", async () => {
    expect(await rendered({ ...MET, metric: "val_loss", direction: "at_most", observed: null, passed: false }))
      .toBe("Criterion missed: validation loss ≤ 0.5, observed not a finite number. Criterion stored before the run as 0123456789ab.");
  });

  it("shows nothing when no criterion was declared", async () => {
    expect(await rendered(null)).toBe("");
  });
});
