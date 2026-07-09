// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import type { ModelSummary } from "./api/client";

export interface ComparisonRow {
  label: string;
  values: string[];
}

const EVIDENCE: Record<number, string> = {
  3: "T3 verified",
  2: "T2 curated",
  1: "T1 declared",
  0: "T0",
};

/** Build the attribute rows for a side-by-side comparison of selected models.
 *
 * Each row is one attribute (family, evidence, maturity, structure, citation);
 * the values align positionally with the supplied models so a table can render
 * them column-per-model.
 */
export function buildComparisonRows(models: ModelSummary[]): ComparisonRow[] {
  if (models.length === 0) return [];
  return [
    { label: "family", values: models.map((m) => m.family || "—") },
    { label: "evidence", values: models.map((m) => EVIDENCE[m.tier] ?? `T${m.tier}`) },
    { label: "maturity", values: models.map((m) => m.maturity || "—") },
    { label: "state vars", values: models.map((m) => String(m.n_state_vars)) },
    { label: "params", values: models.map((m) => String(m.n_params)) },
    { label: "doi", values: models.map((m) => m.provenance?.doi || "—") },
  ];
}
