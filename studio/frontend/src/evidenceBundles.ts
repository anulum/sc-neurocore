// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio evidence bundle helpers
import type {
  StudioAuditStatus,
  StudioEvidenceBundleResponse,
  StudioJobListResponse,
  StudioJobRecord,
  StudioJobStatus,
  StudioOperatorStatus,
} from "./api/client";
import { downloadBrowserArtefact } from "./browserArtefactDownload";

export type EvidenceBundleSurface = "admin" | "project" | "compile" | "synthesis";
export type ScopedEvidenceBundleSurface = Exclude<EvidenceBundleSurface, "admin">;

export interface EvidenceBundleSlots {
  compileEvidenceBundle: StudioEvidenceBundleResponse | null;
  projectEvidenceBundle: StudioEvidenceBundleResponse | null;
  synthesisEvidenceBundle: StudioEvidenceBundleResponse | null;
}

export interface EvidenceBundleDownloadSlots extends EvidenceBundleSlots {
  evidenceBundle: StudioEvidenceBundleResponse | null;
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

export type EvidenceBundleDownloadErrorKey =
  | "compileEvidenceBundleError"
  | "evidenceBundleError"
  | "projectEvidenceBundleError"
  | "synthesisEvidenceBundleError";

export interface EvidenceBundleDownloadSelection {
  bundle: StudioEvidenceBundleResponse | null;
  error: EvidenceBundleDownloadErrorKey;
}

export type EvidenceBundleArtefactDownloader = (payload: Blob, relativePath: string) => void;

export interface EvidenceBundleArtifactDownloadUnavailablePlan {
  available: false;
  statePatch: EvidenceBundleDownloadStatePatch;
}

export interface EvidenceBundleArtifactDownloadReadyPlan {
  available: true;
  failureState: (error: unknown) => EvidenceBundleDownloadStatePatch;
  jobId: string;
  relativePath: string;
  startState: EvidenceBundleDownloadStatePatch;
  writePayload: (payload: Blob, downloader?: EvidenceBundleArtefactDownloader) => void;
}

export type EvidenceBundleArtifactDownloadPlan =
  | EvidenceBundleArtifactDownloadReadyPlan
  | EvidenceBundleArtifactDownloadUnavailablePlan;

export interface EvidenceBundleOperatorRefreshPatch {
  auditStatus: StudioAuditStatus;
  jobRecords: StudioJobRecord[];
  jobStatus: StudioJobStatus;
  operatorStatus: StudioOperatorStatus;
}

export interface AdminEvidenceBundleLoadingStatePatch {
  evidenceBundleError: null;
  evidenceBundleLoading: true;
}

export interface AdminEvidenceBundleCreatedStatePatch
  extends EvidenceBundleOperatorRefreshPatch {
  evidenceBundle: StudioEvidenceBundleResponse;
  evidenceBundleError: null;
  evidenceBundleLoading: false;
}

export interface AdminEvidenceBundleFailureStatePatch {
  evidenceBundleError: string;
  evidenceBundleLoading: false;
}

export interface EvidenceBundleSurfaceStatePatch
  extends Partial<Record<EvidenceBundleSurfaceKeys["bundle"], StudioEvidenceBundleResponse>> {
  compileEvidenceBundleError?: string | null;
  compileEvidenceBundleLoading?: boolean;
  projectEvidenceBundleError?: string | null;
  projectEvidenceBundleLoading?: boolean;
  synthesisEvidenceBundleError?: string | null;
  synthesisEvidenceBundleLoading?: boolean;
}

export type ScopedEvidenceBundleCreatedStatePatch =
  EvidenceBundleOperatorRefreshPatch & EvidenceBundleSurfaceStatePatch;

export type EvidenceBundleDownloadStatePatch =
  Partial<Record<EvidenceBundleDownloadErrorKey, string | null>>;

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

export function evidenceBundleDownloadSelection(
  surface: EvidenceBundleSurface,
  slots: EvidenceBundleDownloadSlots,
): EvidenceBundleDownloadSelection {
  if (surface === "admin") {
    return {
      bundle: slots.evidenceBundle,
      error: "evidenceBundleError",
    };
  }
  const keys = evidenceBundleSurfaceKeys(surface);
  return {
    bundle: slots[keys.bundle],
    error: keys.error,
  };
}

export function adminEvidenceBundleLoadingState(): AdminEvidenceBundleLoadingStatePatch {
  return {
    evidenceBundleError: null,
    evidenceBundleLoading: true,
  };
}

export function adminEvidenceBundleCreatedState(
  evidenceBundle: StudioEvidenceBundleResponse,
  operatorStatus: StudioOperatorStatus,
  jobList: StudioJobListResponse,
): AdminEvidenceBundleCreatedStatePatch {
  return {
    ...operatorRefreshState(operatorStatus, jobList),
    evidenceBundle,
    evidenceBundleError: null,
    evidenceBundleLoading: false,
  };
}

export function adminEvidenceBundleFailureState(
  error: unknown,
): AdminEvidenceBundleFailureStatePatch {
  return {
    evidenceBundleError: errorMessage(error, "Evidence bundle export failed"),
    evidenceBundleLoading: false,
  };
}

export function scopedEvidenceBundleLoadingState(
  surface: ScopedEvidenceBundleSurface,
): EvidenceBundleSurfaceStatePatch {
  const keys = evidenceBundleSurfaceKeys(surface);
  return {
    [keys.error]: null,
    [keys.loading]: true,
  };
}

export function scopedEvidenceBundleCreatedState(
  surface: ScopedEvidenceBundleSurface,
  evidenceBundle: StudioEvidenceBundleResponse,
  operatorStatus: StudioOperatorStatus,
  jobList: StudioJobListResponse,
): ScopedEvidenceBundleCreatedStatePatch {
  const keys = evidenceBundleSurfaceKeys(surface);
  return {
    ...operatorRefreshState(operatorStatus, jobList),
    [keys.bundle]: evidenceBundle,
    [keys.error]: null,
    [keys.loading]: false,
  };
}

export function scopedEvidenceBundleFailureState(
  surface: ScopedEvidenceBundleSurface,
  error: unknown,
): EvidenceBundleSurfaceStatePatch {
  const keys = evidenceBundleSurfaceKeys(surface);
  return {
    [keys.error]: errorMessage(error, "Evidence bundle export failed"),
    [keys.loading]: false,
  };
}

export function evidenceBundleArtifactUnavailableState(
  surface: EvidenceBundleSurface,
): EvidenceBundleDownloadStatePatch {
  return evidenceBundleDownloadErrorState(
    surface,
    "No evidence bundle is available for artifact download.",
  );
}

export function evidenceBundleArtifactDownloadStartState(
  surface: EvidenceBundleSurface,
): EvidenceBundleDownloadStatePatch {
  return evidenceBundleDownloadErrorState(surface, null);
}

export function evidenceBundleArtifactDownloadFailureState(
  surface: EvidenceBundleSurface,
  error: unknown,
): EvidenceBundleDownloadStatePatch {
  return evidenceBundleDownloadErrorState(
    surface,
    errorMessage(error, "Evidence artefact download failed"),
  );
}

export function evidenceBundleArtifactDownloadPlan(
  surface: EvidenceBundleSurface,
  relativePath: string,
  slots: EvidenceBundleDownloadSlots,
): EvidenceBundleArtifactDownloadPlan {
  const { bundle } = evidenceBundleDownloadSelection(surface, slots);
  if (bundle === null) {
    return {
      available: false,
      statePatch: evidenceBundleArtifactUnavailableState(surface),
    };
  }
  return {
    available: true,
    failureState: (error) => evidenceBundleArtifactDownloadFailureState(surface, error),
    jobId: bundle.job_id,
    relativePath,
    startState: evidenceBundleArtifactDownloadStartState(surface),
    writePayload: (payload, downloader = downloadBrowserArtefact) => {
      downloader(payload, relativePath);
    },
  };
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

function operatorRefreshState(
  operatorStatus: StudioOperatorStatus,
  jobList: StudioJobListResponse,
): EvidenceBundleOperatorRefreshPatch {
  return {
    auditStatus: operatorStatus.audit,
    jobRecords: jobList.jobs,
    jobStatus: operatorStatus.jobs,
    operatorStatus,
  };
}

function evidenceBundleDownloadErrorState(
  surface: EvidenceBundleSurface,
  message: string | null,
): EvidenceBundleDownloadStatePatch {
  const { error } = evidenceBundleDownloadSelection(surface, {
    compileEvidenceBundle: null,
    evidenceBundle: null,
    projectEvidenceBundle: null,
    synthesisEvidenceBundle: null,
  });
  return { [error]: message };
}

function errorMessage(error: unknown, fallbackMessage: string): string {
  return error instanceof Error && error.message.length > 0 ? error.message : fallbackMessage;
}
