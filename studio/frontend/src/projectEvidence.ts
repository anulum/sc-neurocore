// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import type { ProjectSaveResponse } from "./api/client";

export interface ProjectEvidenceModel {
  classification: string;
  name: string;
  projectDigest: string;
  schemaVersion: string;
  stateDigest: string;
}

export function buildProjectEvidenceModel(response: ProjectSaveResponse): ProjectEvidenceModel {
  return {
    classification: response.evidence_classification,
    name: response.name,
    projectDigest: response.project_sha256.slice(0, 12),
    schemaVersion: response.schema_version,
    stateDigest: response.state_sha256.slice(0, 12),
  };
}
