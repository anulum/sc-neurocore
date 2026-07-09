// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { describe, expect, it } from "vitest";

import type { DclsBackendStatus } from "../api/client";
import { backendColor, backendLabel } from "./DclsPanel";

const status = (o: Partial<DclsBackendStatus>): DclsBackendStatus =>
  ({ backend: "x", available: true, live: true, ...o });

describe("DclsPanel backend display", () => {
  it("labels a live bit-exact backend", () => {
    const b = status({ bit_exact: true });
    expect(backendLabel(b)).toBe("bit-exact");
    expect(backendColor(b)).toBe("var(--success)");
  });

  it("flags a live divergent backend as a warning", () => {
    const b = status({ bit_exact: false });
    expect(backendLabel(b)).toBe("DIVERGES");
    expect(backendColor(b)).toBe("var(--warning)");
  });

  it("marks a declared-but-not-run backend as verified offline", () => {
    const b = status({ live: false });
    expect(backendLabel(b)).toBe("offline ✓");
    expect(backendColor(b)).toBe("var(--accent)");
  });

  it("shows an unavailable backend muted", () => {
    const b = status({ available: false });
    expect(backendLabel(b)).toBe("—");
    expect(backendColor(b)).toBe("var(--text-muted)");
  });
});
