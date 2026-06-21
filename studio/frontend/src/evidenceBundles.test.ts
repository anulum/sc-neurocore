// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio evidence bundle helper tests
import { describe, expect, it } from "vitest";

import type { StudioEvidenceBundleResponse, StudioJobRecord } from "./api/client";
import {
  evidenceBundleSurfaceKeys,
  latestSynthesisJobIdWithArtefact,
  selectEvidenceBundleForSurface,
} from "./evidenceBundles";

const bundle: StudioEvidenceBundleResponse = {
  artifact_paths: ["evidence/manifest.json"],
  artifacts: [],
  bundle_id: "seb_project",
  job_id: "sj_bundle",
  manifest: {},
  schema_version: "studio.evidence-bundle.v1",
  summary: {
    artifact_path_count: 1,
    entry_count: 1,
    entry_type_counts: { manifest: 1 },
    evidence_classification_counts: {},
    source_job_count: 0,
    source_job_kind_counts: {},
    source_job_owner_counts: {},
  },
};

function jobRecord(
  jobId: string,
  createdAt: string,
  kind: string,
  artefactPaths: string[],
): StudioJobRecord {
  return {
    artifacts: artefactPaths.map((relativePath) => ({
      relative_path: relativePath,
      sha256: "a".repeat(64),
      size_bytes: 128,
    })),
    created_at_utc: createdAt,
    error: null,
    execution_model: "process",
    finished_at_utc: createdAt,
    job_id: jobId,
    kind,
    owner: "studio",
    request_id: null,
    result: null,
    started_at_utc: createdAt,
    status: "completed",
  };
}

describe("evidence bundle surface helpers", () => {
  it("maps scoped surfaces to store keys", () => {
    expect(evidenceBundleSurfaceKeys("project")).toEqual({
      bundle: "projectEvidenceBundle",
      error: "projectEvidenceBundleError",
      loading: "projectEvidenceBundleLoading",
    });
    expect(evidenceBundleSurfaceKeys("compile")).toEqual({
      bundle: "compileEvidenceBundle",
      error: "compileEvidenceBundleError",
      loading: "compileEvidenceBundleLoading",
    });
    expect(evidenceBundleSurfaceKeys("synthesis")).toEqual({
      bundle: "synthesisEvidenceBundle",
      error: "synthesisEvidenceBundleError",
      loading: "synthesisEvidenceBundleLoading",
    });
  });

  it("selects the evidence bundle for a scoped surface", () => {
    expect(selectEvidenceBundleForSurface("project", {
      compileEvidenceBundle: null,
      projectEvidenceBundle: bundle,
      synthesisEvidenceBundle: null,
    })).toBe(bundle);
  });

  it("chooses the newest synthesis job carrying the requested artefact", () => {
    const jobs = [
      jobRecord("sj_old", "2026-06-21T10:00:00Z", "synthesis", [
        "synthesis/multi-target-result.json",
      ]),
      jobRecord("sj_new", "2026-06-21T11:00:00Z", "synthesis", [
        "synthesis/multi-target-result.json",
      ]),
      jobRecord("sj_compile", "2026-06-21T12:00:00Z", "compiler", [
        "synthesis/multi-target-result.json",
      ]),
    ];

    expect(latestSynthesisJobIdWithArtefact(
      jobs,
      "synthesis/multi-target-result.json",
    )).toBe("sj_new");
  });

  it("returns null when no synthesis job carries the requested artefact", () => {
    expect(latestSynthesisJobIdWithArtefact([
      jobRecord("sj_compile", "2026-06-21T12:00:00Z", "compiler", [
        "compiler/result.json",
      ]),
    ], "synthesis/result.json")).toBeNull();
  });
});
