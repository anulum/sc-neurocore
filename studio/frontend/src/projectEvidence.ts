// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import type { ProjectSaveResponse } from "./api/client";

/** What the evidence strip states about a saved project. */
export interface ProjectEvidenceModel {
  classification: string;
  name: string;
  projectDigest: string;
  schemaVersion: string;
  stateDigest: string;
}

/**
 * Describe a saved project for its evidence strip.
 *
 * The digests are shown as twelve characters: enough to compare two saves
 * by eye, with the full values in the response for a real check.
 *
 * @param response - The save's response.
 * @returns What the strip should state.
 */
export function buildProjectEvidenceModel(response: ProjectSaveResponse): ProjectEvidenceModel {
  return {
    classification: response.evidence_classification,
    name: response.name,
    projectDigest: response.project_sha256.slice(0, 12),
    schemaVersion: response.schema_version,
    stateDigest: response.state_sha256.slice(0, 12),
  };
}
