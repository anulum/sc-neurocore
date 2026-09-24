// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio model silicon capability row tests

import { describe, expect, it } from "vitest";

import type { ModelCapabilities } from "./api/client";
import { capabilityRows } from "./modelCapabilities";

const disabled = (reason: string) => ({ enabled: false, reason });

describe("silicon capability rows", () => {
  it("states what each enabled operation runs, in pipeline order", () => {
    const capabilities: ModelCapabilities = {
      schema_version: "sc-neurocore.studio.model-capabilities.v1",
      model: "AdaptiveThresholdIFNeuron",
      operations: {
        compile: { enabled: true, reason: null, integrators: ["map"], q_formats: ["Q8.8", "Q16.16"] },
        cosimulate: {
          enabled: true,
          reason: null,
          combinations: [
            { integrator: "map", q_format: "Q8.8", mirrored: true },
            { integrator: "map", q_format: "Q16.16", mirrored: false },
          ],
        },
        synthesise: { enabled: true, reason: null, targets: ["ice40", "ecp5"] },
        place_and_route: { enabled: true, reason: null, targets: ["ice40"] },
        formal: {
          ...disabled("the Studio has no route that runs a formal job"),
          catalogue_job: {
            module: "sc_adaptive_threshold_if",
            claim: "bounded safety",
            q_format: "Q8.8",
            depth: 8,
            properties: [],
            not_established: [],
          },
        },
      },
    };

    expect(capabilityRows(capabilities)).toEqual([
      { label: "compile", enabled: true, detail: "map · Q8.8, Q16.16" },
      { label: "co-simulate", enabled: true, detail: "map Q8.8" },
      { label: "synthesise", enabled: true, detail: "ice40, ecp5" },
      { label: "place & route", enabled: true, detail: "ice40" },
      {
        label: "formal",
        enabled: false,
        detail: "the Studio has no route that runs a formal job "
          + "(catalogue job sc_adaptive_threshold_if: bounded safety to depth 8)",
      },
    ]);
  });

  it("gives every disabled operation its reason", () => {
    const capabilities: ModelCapabilities = {
      schema_version: "sc-neurocore.studio.model-capabilities.v1",
      model: "SCScaledResetAdaptiveIFNeuron",
      operations: {
        compile: { enabled: true, reason: null },
        cosimulate: disabled("no bit-true C kernel mirrors this model's RTL for any offered integrator"),
        synthesise: disabled("synthesis runs only on RTL whose co-simulation was bit-exact"),
        place_and_route: { enabled: false, reason: null },
        formal: { ...disabled("the Studio has no route that runs a formal job"), catalogue_job: null },
      },
    };

    const rows = capabilityRows(capabilities);
    expect(rows[0]).toEqual({ label: "compile", enabled: true, detail: " · " });
    expect(rows[1]?.detail).toBe("no bit-true C kernel mirrors this model's RTL for any offered integrator");
    expect(rows[3]?.detail).toBe("unavailable");
    expect(rows[4]?.detail).toBe("the Studio has no route that runs a formal job");
  });
});
