// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { describe, expect, it } from "vitest";

import type { ModelSummary } from "./api/client";
import { buildComparisonRows } from "./modelComparison";

const summary = (o: Partial<ModelSummary>): ModelSummary =>
  ({
    name: "X", module: "x", category: "c", tier: 2, evidence_kind: "curated",
    category_slug: "c", category_source: "declared", family: "Fam", maturity: "validated",
    biophysical_detail: "", n_state_vars: 2, n_params: 5, state_var_names: [],
    docstring: "", display_name: "X", provenance: null,
    ...o,
  }) as ModelSummary;

describe("buildComparisonRows", () => {
  it("returns no rows for an empty selection", () => {
    expect(buildComparisonRows([])).toEqual([]);
  });

  it("aligns each attribute's values positionally with the models", () => {
    const rows = buildComparisonRows([
      summary({ name: "A", tier: 3, n_params: 7, provenance: { authors: [], year: null, doi: "10.1/a", paper_title: "", url: "", citeable: true } }),
      summary({ name: "B", tier: 1, family: "Other", maturity: "experimental" }),
    ]);
    const byLabel = Object.fromEntries(rows.map((r) => [r.label, r.values]));
    expect(byLabel["evidence"]).toEqual(["T3 verified", "T1 declared"]);
    expect(byLabel["params"]).toEqual(["7", "5"]);
    expect(byLabel["doi"]).toEqual(["10.1/a", "—"]);
    expect(byLabel["family"]).toEqual(["Fam", "Other"]);
  });
});
