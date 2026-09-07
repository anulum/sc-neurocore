// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import type { ModelSummary } from "./api/client";

/** One attribute of the comparison, and its value per model. */
export interface ComparisonRow {
  label: string;
  values: string[];
}

/**
 * What each evidence tier is called. A tier with no name falls back to its
 * number rather than to a blank: an unnamed tier is still information.
 */
const EVIDENCE: Record<number, string> = {
  3: "T3 verified",
  2: "T2 curated",
  1: "T1 declared",
  0: "T0",
};

/**
 * Show a field, or a dash where there is nothing to show.
 *
 * This is `||` behaviour rather than `??` on purpose, and it is a function so
 * the reason has somewhere to live: these fields arrive as empty strings when
 * the catalogue has no value, and `??` would put an empty cell in the table
 * where the reader needs to see that nothing is recorded.
 *
 * @param value - The field, if the catalogue carries one.
 * @returns The value, or an em dash.
 */
function presentOrDash(value: string | undefined): string {
  return value !== undefined && value.length > 0 ? value : "—";
}

/**
 * Build the rows of a side-by-side model comparison.
 *
 * Each row is one attribute and its values line up positionally with the
 * models, so a table renders one column per model without re-deriving
 * anything.
 *
 * @param models - The models being compared, in column order.
 * @returns The rows, or none at all for an empty selection: a table of headers
 *   with no values reads as a failed load.
 */
export function buildComparisonRows(models: ModelSummary[]): ComparisonRow[] {
  if (models.length === 0) return [];
  return [
    { label: "family", values: models.map((m) => presentOrDash(m.family)) },
    { label: "evidence", values: models.map((m) => EVIDENCE[m.tier] ?? `T${m.tier}`) },
    { label: "maturity", values: models.map((m) => presentOrDash(m.maturity)) },
    { label: "state vars", values: models.map((m) => String(m.n_state_vars)) },
    { label: "params", values: models.map((m) => String(m.n_params)) },
    { label: "doi", values: models.map((m) => presentOrDash(m.provenance?.doi)) },
  ];
}
