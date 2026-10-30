// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Catalogue candidate response custody

/** A server-owned copy whose JSON text retains declaration order and integer tokens. */
export interface CatalogueCandidate {
  schema_version: "studio.candidate-catalogue-start.v1";
  parent_model: string;
  parent_schema: string;
  parent_schema_sha256: string;
  candidate_text: string;
}

/**
 * Check the response identity without parsing and rewriting its candidate text.
 *
 * @param value - Actual JSON response from the catalogue authoring route.
 * @param modelName - Exact catalogue model requested by this action.
 * @returns Checked source identity and untouched candidate text.
 * @throws {Error} When the response is incomplete or names another model.
 */
export function parseCatalogueCandidate(value: unknown, modelName: string): CatalogueCandidate {
  if (typeof value !== "object" || value === null) {
    throw new Error("The catalogue draft response is invalid.");
  }
  const row = value as Record<string, unknown>;
  if (row.schema_version !== "studio.candidate-catalogue-start.v1"
    || row.parent_model !== modelName
    || typeof row.parent_schema !== "string" || row.parent_schema.length === 0
    || typeof row.parent_schema_sha256 !== "string" || !/^[0-9a-f]{64}$/u.test(row.parent_schema_sha256)
    || typeof row.candidate_text !== "string" || row.candidate_text.length === 0) {
    throw new Error("The catalogue draft response is invalid.");
  }
  return {
    schema_version: row.schema_version,
    parent_model: modelName,
    parent_schema: row.parent_schema,
    parent_schema_sha256: row.parent_schema_sha256,
    candidate_text: row.candidate_text,
  };
}
