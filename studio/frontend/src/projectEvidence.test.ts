// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { describe, expect, it } from "vitest";

import type { ProjectSaveResponse } from "./api/client";
import { buildProjectEvidenceModel } from "./projectEvidence";

const response: ProjectSaveResponse = {
  evidence_classification: "project_workspace",
  name: "saved-network",
  project_sha256: "a".repeat(64),
  saved_at: 1_782_000_000,
  schema_version: "studio.project-save.v1",
  state_sha256: "b".repeat(64),
  version: "3.14.0",
};

describe("project evidence model", () => {
  it("summarizes path-free project save evidence", () => {
    expect(buildProjectEvidenceModel(response)).toEqual({
      classification: "project_workspace",
      name: "saved-network",
      projectDigest: "aaaaaaaaaaaa",
      schemaVersion: "studio.project-save.v1",
      stateDigest: "bbbbbbbbbbbb",
    });
  });
});
