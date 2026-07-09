// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { describe, expect, it } from "vitest";

import type { ModelBackendSupport } from "../api/client";
import { orderedBackends } from "./BackendMatrix";

const b = (name: string, status: string, parity: string): ModelBackendSupport =>
  ({ name, status, parity }) as ModelBackendSupport;

describe("orderedBackends", () => {
  it("keeps only implemented backends, Python reference first", () => {
    const result = orderedBackends([
      b("mojo", "implemented", "ulp-bounded"),
      b("rust", "planned", "exact"),
      b("python", "implemented", "exact"),
      b("julia", "implemented", "exact"),
    ]);
    expect(result.map((x) => x.name)).toEqual(["python", "julia", "mojo"]);
  });

  it("returns a single backend for Python-only models", () => {
    const result = orderedBackends([b("python", "implemented", "exact")]);
    expect(result.map((x) => x.name)).toEqual(["python"]);
  });
});
