// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio evidence bundle helpers

/**
 * Evidence bundles: which surface a bundle is about, and how one is fetched.
 *
 * Four surfaces gather evidence — the admin panel and three scoped ones
 * (project, compile, synthesis) — and each keeps its own bundle, its own
 * loading flag and its own error. They are deliberately not one shared slot: a
 * failed synthesis export must not blank the compile bundle a reader is
 * looking at, and two exports may be in flight at once.
 *
 * The download half is a *plan* rather than an action. Deciding which job holds
 * an artefact, and refusing when none does, is separable from fetching it — so
 * the decision is tested without a network and the fetch has nothing to decide.
 */

/**
 * Evidence bundles: which surface a bundle is about, and how one is fetched.
 *
 * Four surfaces gather evidence — the admin panel and three scoped ones
 * (project, compile, synthesis) — and each keeps its own bundle, its own
 * loading flag and its own error. They are deliberately not one shared slot: a
 * failed synthesis export must not blank the compile bundle a reader is
 * looking at, and two exports may be in flight at once.
 *
 * The download half is a *plan* rather than an action. Deciding which job holds
 * an artefact, and refusing when none does, is separable from fetching it — so
 * the decision is tested without a network and the fetch has nothing to decide.
 */
import type {
  StudioAuditStatus,
  StudioEvidenceBundleResponse,
  StudioJobListResponse,
  StudioJobRecord,
  StudioJobStatus,
  StudioOperatorStatus,
} from "./api/client";
import { downloadBrowserArtefact } from "./browserArtefactDownload";

/** Which part of the Studio a bundle is about. */
export type EvidenceBundleSurface = "admin" | "project" | "compile" | "synthesis";
/**
 * The three surfaces that keep their own bundle.
 *
 * The admin panel is excluded: it gathers across surfaces rather than being
 * one of them.
 */
export type ScopedEvidenceBundleSurface = Exclude<EvidenceBundleSurface, "admin">;

/** The store fields the four surfaces keep their bundles in. */
export interface EvidenceBundleSlots {
  compileEvidenceBundle: StudioEvidenceBundleResponse | null;
  projectEvidenceBundle: StudioEvidenceBundleResponse | null;
  synthesisEvidenceBundle: StudioEvidenceBundleResponse | null;
}

/** The bundle slots plus what a download needs to find its artefact. */
export interface EvidenceBundleDownloadSlots extends EvidenceBundleSlots {
  evidenceBundle: StudioEvidenceBundleResponse | null;
}

/** One surface's three store keys: its bundle, its loading flag, its error. */
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

/** Which surface's error a failed download is written to. */
export type EvidenceBundleDownloadErrorKey =
  | "compileEvidenceBundleError"
  | "evidenceBundleError"
  | "projectEvidenceBundleError"
  | "synthesisEvidenceBundleError";

/** The bundle a download would read, and where its errors go. */
export interface EvidenceBundleDownloadSelection {
  bundle: StudioEvidenceBundleResponse | null;
  error: EvidenceBundleDownloadErrorKey;
}

/** How a fetched artefact reaches the reader's disk. */
export type EvidenceBundleArtefactDownloader = (payload: Blob, relativePath: string) => void;

/**
 * There is nothing to download, and why.
 *
 * A refusal carries its reason so the panel can say which of the several
 * preconditions was not met.
 */
export interface EvidenceBundleArtifactDownloadUnavailablePlan {
  available: false;
  statePatch: EvidenceBundleDownloadStatePatch;
}

/** Everything a download needs: the job, the path, and where to report. */
export interface EvidenceBundleArtifactDownloadReadyPlan {
  available: true;
  failureState: (error: unknown) => EvidenceBundleDownloadStatePatch;
  jobId: string;
  relativePath: string;
  startState: EvidenceBundleDownloadStatePatch;
  writePayload: (payload: Blob, downloader?: EvidenceBundleArtefactDownloader) => void;
}

/** A download that can proceed, or a refusal that says why not. */
export type EvidenceBundleArtifactDownloadPlan =
  | EvidenceBundleArtifactDownloadReadyPlan
  | EvidenceBundleArtifactDownloadUnavailablePlan;

/** The operator view, refreshed because a bundle changed it. */
export interface EvidenceBundleOperatorRefreshPatch {
  auditStatus: StudioAuditStatus;
  jobRecords: StudioJobRecord[];
  jobStatus: StudioJobStatus;
  operatorStatus: StudioOperatorStatus;
}

/** The admin bundle is being gathered. */
export interface AdminEvidenceBundleLoadingStatePatch {
  evidenceBundleError: null;
  evidenceBundleLoading: true;
}

/** The admin bundle arrived, with the operator view refreshed. */
export interface AdminEvidenceBundleCreatedStatePatch
  extends EvidenceBundleOperatorRefreshPatch {
  evidenceBundle: StudioEvidenceBundleResponse;
  evidenceBundleError: null;
  evidenceBundleLoading: false;
}

/** The admin bundle failed, with the message to show. */
export interface AdminEvidenceBundleFailureStatePatch {
  evidenceBundleError: string;
  evidenceBundleLoading: false;
}

/** One scoped surface's bundle, loading flag and error. */
export interface EvidenceBundleSurfaceStatePatch
  extends Partial<Record<EvidenceBundleSurfaceKeys["bundle"], StudioEvidenceBundleResponse>> {
  compileEvidenceBundleError?: string | null;
  compileEvidenceBundleLoading?: boolean;
  projectEvidenceBundleError?: string | null;
  projectEvidenceBundleLoading?: boolean;
  synthesisEvidenceBundleError?: string | null;
  synthesisEvidenceBundleLoading?: boolean;
}

/** A scoped bundle arrived, with the operator view refreshed. */
export type ScopedEvidenceBundleCreatedStatePatch =
  EvidenceBundleOperatorRefreshPatch & EvidenceBundleSurfaceStatePatch;

/** A download started, failed, or was refused. */
export type EvidenceBundleDownloadStatePatch =
  Partial<Record<EvidenceBundleDownloadErrorKey, string | null>>;

/** Each scoped surface's three store keys. */
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

/**
 * The store keys one scoped surface uses.
 *
 * @param surface - The surface.
 * @returns Its bundle, loading and error keys.
 */
export function evidenceBundleSurfaceKeys(
  surface: ScopedEvidenceBundleSurface,
): EvidenceBundleSurfaceKeys {
  return evidenceBundleKeys[surface];
}

/**
 * The bundle one surface currently holds, if any.
 *
 * @param surface - The surface.
 * @param slots - The store's bundle fields.
 * @returns The bundle, or `null`.
 */
export function selectEvidenceBundleForSurface(
  surface: ScopedEvidenceBundleSurface,
  slots: EvidenceBundleSlots,
): StudioEvidenceBundleResponse | null {
  return slots[evidenceBundleKeys[surface].bundle];
}

/**
 * The bundle a download would read and where its errors belong.
 *
 * @param surface - The surface.
 * @param slots - The store's bundle fields.
 * @returns The selection.
 */
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

/**
 * Mark the admin bundle as being gathered.
 *
 * @returns The patch.
 */
export function adminEvidenceBundleLoadingState(): AdminEvidenceBundleLoadingStatePatch {
  return {
    evidenceBundleError: null,
    evidenceBundleLoading: true,
  };
}

/**
 * Take a finished admin bundle into the store.
 *
 * @param evidenceBundle - The bundle.
 * @param operatorStatus - The refreshed operator status.
 * @param jobList - The refreshed job list.
 * @returns The patch.
 */
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

/**
 * Report a failed admin bundle.
 *
 * @param error - Whatever was thrown or rejected.
 * @returns The patch.
 */
export function adminEvidenceBundleFailureState(
  error: unknown,
): AdminEvidenceBundleFailureStatePatch {
  return {
    evidenceBundleError: errorMessage(error, "Evidence bundle export failed"),
    evidenceBundleLoading: false,
  };
}

/**
 * Mark one scoped surface's bundle as being gathered.
 *
 * @param surface - The surface.
 * @returns The patch.
 */
export function scopedEvidenceBundleLoadingState(
  surface: ScopedEvidenceBundleSurface,
): EvidenceBundleSurfaceStatePatch {
  const keys = evidenceBundleSurfaceKeys(surface);
  return {
    [keys.error]: null,
    [keys.loading]: true,
  };
}

/**
 * Take a finished scoped bundle into the store.
 *
 * @param surface - The surface.
 * @param evidenceBundle - The bundle.
 * @param operatorStatus - The refreshed operator status.
 * @param jobList - The refreshed job list.
 * @returns The patch.
 */
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

/**
 * Report a failed scoped bundle, on that surface only.
 *
 * @param surface - The surface.
 * @param error - Whatever was thrown or rejected.
 * @returns The patch.
 */
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

/**
 * Say that a surface has no artefact to download.
 *
 * @param surface - The surface.
 * @returns The patch.
 */
export function evidenceBundleArtifactUnavailableState(
  surface: EvidenceBundleSurface,
): EvidenceBundleDownloadStatePatch {
  return evidenceBundleDownloadErrorState(
    surface,
    "No evidence bundle is available for artifact download.",
  );
}

/**
 * Mark an artefact download as begun.
 *
 * @param surface - The surface.
 * @returns The patch.
 */
export function evidenceBundleArtifactDownloadStartState(
  surface: EvidenceBundleSurface,
): EvidenceBundleDownloadStatePatch {
  return evidenceBundleDownloadErrorState(surface, null);
}

/**
 * Report a failed artefact download.
 *
 * @param surface - The surface.
 * @param error - Whatever was thrown or rejected.
 * @returns The patch.
 */
export function evidenceBundleArtifactDownloadFailureState(
  surface: EvidenceBundleSurface,
  error: unknown,
): EvidenceBundleDownloadStatePatch {
  return evidenceBundleDownloadErrorState(
    surface,
    errorMessage(error, "Evidence artefact download failed"),
  );
}

/**
 * Decide whether an artefact can be downloaded, and from which job.
 *
 * A plan rather than an action: the decision is the part with rules in it, and
 * separating it means those rules are checked without a network.
 *
 * @param surface - The surface asking.
 * @param relativePath - The artefact within the job.
 * @param slots - The store fields the decision reads.
 * @returns A plan that can proceed, or a refusal that says why not.
 */
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

/**
 * Read the exact completed synthesis job from its response receipt.
 *
 * Global job order cannot identify this response. Missing or malformed legacy
 * receipts leave export unavailable without inventing a replacement job.
 *
 * @param receipt - Untrusted response metadata to validate.
 * @param artefactPath - The artefact to look for.
 * @returns The job's identifier, or `null` when none holds it.
 */
export function synthesisJobIdFromReceipt(
  receipt: unknown,
  artefactPath: string,
): string | null {
  if (receipt === null || typeof receipt !== "object"
    || !("schema_version" in receipt) || receipt.schema_version !== "studio.job-receipt.v1"
    || !("status" in receipt) || receipt.status !== "completed"
    || !("kind" in receipt) || receipt.kind !== "synthesis"
    || !("job_id" in receipt) || typeof receipt.job_id !== "string"
    || !/^sj_[A-Za-z0-9_-]+$/.test(receipt.job_id)
    || !("artifacts" in receipt) || !Array.isArray(receipt.artifacts)) return null;
  const matches = receipt.artifacts.filter((artifact: unknown) => artifact !== null
    && typeof artifact === "object" && "relative_path" in artifact
    && artifact.relative_path === artefactPath) as unknown[];
  if (matches.length !== 1) return null;
  const artifact = matches[0];
  if (artifact === null || typeof artifact !== "object"
    || !("sha256" in artifact) || typeof artifact.sha256 !== "string"
    || !/^[a-f0-9]{64}$/.test(artifact.sha256)
    || !("size_bytes" in artifact) || typeof artifact.size_bytes !== "number"
    || !Number.isSafeInteger(artifact.size_bytes) || artifact.size_bytes < 0) return null;
  return receipt.job_id;
}

/**
 * The operator view, refreshed after a bundle changed it.
 *
 * @param operatorStatus - The refreshed status.
 * @param jobList - The refreshed job list.
 * @returns The patch.
 */
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

/**
 * Write a download error to the surface that asked for it.
 *
 * @param surface - The surface.
 * @param message - The message to show.
 * @returns The patch.
 */
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

/**
 * The message to show for a thrown value.
 *
 * @param error - Whatever was thrown.
 * @param fallbackMessage - What to show when it carries no message.
 * @returns The message.
 */
function errorMessage(error: unknown, fallbackMessage: string): string {
  return error instanceof Error && error.message.length > 0 ? error.message : fallbackMessage;
}
