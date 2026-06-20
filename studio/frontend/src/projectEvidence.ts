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
