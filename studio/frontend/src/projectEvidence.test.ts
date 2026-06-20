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
