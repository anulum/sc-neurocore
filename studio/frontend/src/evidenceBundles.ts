// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio evidence bundle helpers
import type { StudioEvidenceBundleResponse, StudioJobRecord } from "./api/client";

export type EvidenceBundleSurface = "admin" | "project" | "compile" | "synthesis";
export type ScopedEvidenceBundleSurface = Exclude<EvidenceBundleSurface, "admin">;

export interface EvidenceBundleSlots {
  compileEvidenceBundle: StudioEvidenceBundleResponse | null;
  projectEvidenceBundle: StudioEvidenceBundleResponse | null;
  synthesisEvidenceBundle: StudioEvidenceBundleResponse | null;
}

export interface EvidenceBundleSurfaceKeys {
  bundle: keyof EvidenceBundleSlots;
  error:
    | "compileEvidenceBundleError"
    | "projectEvidenceBundleError"
    | "synthesisEvidenceBundleError";
  loading:
    | "compileEvidenceBundleLoading"
    | "projectEvidenceBundleLoading"
    | "synthesisEvidenceBundleLoading";
}

const evidenceBundleKeys: Record<ScopedEvidenceBundleSurface, EvidenceBundleSurfaceKeys> = {
  compile: {
    bundle: "compileEvidenceBundle",
    error: "compileEvidenceBundleError",
    loading: "compileEvidenceBundleLoading",
  },
  project: {
    bundle: "projectEvidenceBundle",
    error: "projectEvidenceBundleError",
    loading: "projectEvidenceBundleLoading",
  },
  synthesis: {
    bundle: "synthesisEvidenceBundle",
    error: "synthesisEvidenceBundleError",
    loading: "synthesisEvidenceBundleLoading",
  },
};

export function evidenceBundleSurfaceKeys(
  surface: ScopedEvidenceBundleSurface,
): EvidenceBundleSurfaceKeys {
  return evidenceBundleKeys[surface];
}

export function selectEvidenceBundleForSurface(
  surface: ScopedEvidenceBundleSurface,
  slots: EvidenceBundleSlots,
): StudioEvidenceBundleResponse | null {
  return slots[evidenceBundleKeys[surface].bundle];
}

export function latestSynthesisJobIdWithArtefact(
  jobs: StudioJobRecord[],
  artefactPath: string,
): string | null {
  const records = jobs
    .filter((job) =>
      job.kind === "synthesis"
      && job.artifacts.some((artifact) => artifact.relative_path === artefactPath),
    )
    .sort((left, right) => right.created_at_utc.localeCompare(left.created_at_utc));
  return records[0]?.job_id ?? null;
}
