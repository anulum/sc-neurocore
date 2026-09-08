// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio synthesis store state helpers

/**
 * Synthesis state, and the operator refresh that follows every run.
 *
 * A synthesis run changes what the operator view should say — a new job, new
 * artefacts — so the completion patches carry the refreshed operator status
 * and job list with them rather than leaving a second round trip to be
 * remembered at each call site.
 */
import type {
  MultiTargetResult,
  StudioAuditStatus,
  StudioJobListResponse,
  StudioJobRecord,
  StudioJobStatus,
  StudioOperatorStatus,
  SynthEstimate,
  SynthResult,
  SynthToolInfo,
} from "./api/client";
import { synthesisJobIdFromReceipt } from "./evidenceBundles";

/** The operator status and job list, refreshed after a run changed them. */
export interface SynthesisOperatorRefreshPatch {
  auditStatus: StudioAuditStatus;
  jobRecords: StudioJobRecord[];
  jobStatus: StudioJobStatus;
  operatorStatus: StudioOperatorStatus;
}

/** A synthesis has begun. */
export interface SynthesisRunStartStatePatch {
  activeTab: "synth";
  error: null;
  isSimulating: true;
  latestSynthesisJobId: null;
  multiTargetResult: null;
  synthesisEvidenceBundle: null;
  synthesisEvidenceBundleError: null;
}

/** An all-target synthesis has begun. */
export interface MultiTargetSynthesisRunStartStatePatch {
  activeTab: "synth";
  error: null;
  isSimulating: true;
  latestMultiTargetSynthesisJobId: null;
  synthResult: null;
  synthesisEvidenceBundle: null;
  synthesisEvidenceBundleError: null;
}

/** A synthesis finished, with the operator view refreshed. */
export interface SynthesisRunCompletedStatePatch
  extends SynthesisOperatorRefreshPatch {
  isSimulating: false;
  latestSynthesisJobId: string | null;
  synthResult: SynthResult;
}

/** An all-target synthesis finished, with the operator view refreshed. */
export interface MultiTargetSynthesisRunCompletedStatePatch
  extends SynthesisOperatorRefreshPatch {
  isSimulating: false;
  latestMultiTargetSynthesisJobId: string | null;
  multiTargetResult: MultiTargetResult;
}

/** A synthesis failed, with the message to show. */
export interface SynthesisFailureStatePatch {
  error: string;
  isSimulating: false;
}

/** A message to show without claiming the run ended. */
export interface SynthesisErrorStatePatch {
  error: string;
}

/** A resource estimate arrived; an estimate, not a measurement. */
export interface SynthesisEstimateLoadedStatePatch {
  synthEstimate: SynthEstimate;
}

/** The chosen target device changed. */
export interface SynthesisTargetStatePatch {
  latestMultiTargetSynthesisJobId: null;
  latestSynthesisJobId: null;
  multiTargetResult: null;
  synthEstimate: null;
  synthResult: null;
  synthTarget: string;
  synthesisEvidenceBundle: null;
  synthesisEvidenceBundleError: null;
}

/** Which synthesis tools this deployment can actually reach. */
export interface SynthesisToolStatusLoadedStatePatch {
  toolsAvailable: Record<string, SynthToolInfo>;
}

/**
 * Mark a synthesis as begun.
 *
 * @returns The patch.
 */
export function synthesisRunStartState(): SynthesisRunStartStatePatch {
  return {
    activeTab: "synth",
    error: null,
    isSimulating: true,
    latestSynthesisJobId: null,
    multiTargetResult: null,
    synthesisEvidenceBundle: null,
    synthesisEvidenceBundleError: null,
  };
}

/**
 * Mark an all-target synthesis as begun.
 *
 * @returns The patch.
 */
export function multiTargetSynthesisRunStartState(): MultiTargetSynthesisRunStartStatePatch {
  return {
    activeTab: "synth",
    error: null,
    isSimulating: true,
    latestMultiTargetSynthesisJobId: null,
    synthResult: null,
    synthesisEvidenceBundle: null,
    synthesisEvidenceBundleError: null,
  };
}

/**
 * Take a finished synthesis into the store, with the operator view refreshed.
 *
 * @param synthResult - The run.
 * @param operatorStatus - The refreshed operator status.
 * @param jobList - The refreshed job list.
 * @param resultArtifactPath - Where the run's result was written.
 * @returns The patch.
 */
export function synthesisRunCompletedState(
  synthResult: SynthResult,
  operatorStatus: StudioOperatorStatus,
  jobList: StudioJobListResponse,
  resultArtifactPath = "synthesis/result.json",
): SynthesisRunCompletedStatePatch {
  return {
    ...synthesisOperatorRefreshState(operatorStatus, jobList),
    isSimulating: false,
    latestSynthesisJobId: synthesisJobIdFromReceipt(
      synthResult.studio_job_receipt,
      resultArtifactPath,
    ),
    synthResult,
  };
}

/**
 * Take a finished all-target synthesis into the store.
 *
 * @param multiTargetResult - The runs, one per target.
 * @param operatorStatus - The refreshed operator status.
 * @param jobList - The refreshed job list.
 * @returns The patch.
 */
export function multiTargetSynthesisRunCompletedState(
  multiTargetResult: MultiTargetResult,
  operatorStatus: StudioOperatorStatus,
  jobList: StudioJobListResponse,
): MultiTargetSynthesisRunCompletedStatePatch {
  return {
    ...synthesisOperatorRefreshState(operatorStatus, jobList),
    isSimulating: false,
    latestMultiTargetSynthesisJobId: synthesisJobIdFromReceipt(
      multiTargetResult.studio_job_receipt,
      "synthesis/multi-target-result.json",
    ),
    multiTargetResult,
  };
}

/**
 * Report a failed synthesis.
 *
 * @param error - Whatever was thrown or rejected.
 * @returns The patch.
 */
export function synthesisFailureState(error: unknown): SynthesisFailureStatePatch {
  return {
    error: synthesisErrorMessage(error, "Synthesis failed"),
    isSimulating: false,
  };
}

/**
 * Show a message without claiming the run ended.
 *
 * @param error - Whatever was thrown.
 * @param fallbackMessage - What to show when it carries no message.
 * @returns The patch.
 */
export function synthesisErrorState(
  error: unknown,
  fallbackMessage: string,
): SynthesisErrorStatePatch {
  return { error: synthesisErrorMessage(error, fallbackMessage) };
}

/**
 * Show a message that did not come from a thrown value.
 *
 * @param message - The message.
 * @returns The patch.
 */
export function synthesisErrorMessageState(message: string): SynthesisErrorStatePatch {
  return { error: message };
}

/**
 * Take a resource estimate into the store.
 *
 * @param synthEstimate - The estimate.
 * @returns The patch.
 */
export function synthesisEstimateLoadedState(
  synthEstimate: SynthEstimate,
): SynthesisEstimateLoadedStatePatch {
  return { synthEstimate };
}

/**
 * Change the target device.
 *
 * @param synthTarget - The device family.
 * @returns The patch.
 */
export function synthesisTargetState(synthTarget: string): SynthesisTargetStatePatch {
  return {
    latestMultiTargetSynthesisJobId: null,
    latestSynthesisJobId: null,
    multiTargetResult: null,
    synthEstimate: null,
    synthResult: null,
    synthTarget,
    synthesisEvidenceBundle: null,
    synthesisEvidenceBundleError: null,
  };
}

/**
 * Take the tool availability into the store.
 *
 * @param toolsAvailable - Each tool, and whether it can be reached.
 * @returns The patch.
 */
export function synthesisToolStatusLoadedState(
  toolsAvailable: Record<string, SynthToolInfo>,
): SynthesisToolStatusLoadedStatePatch {
  return { toolsAvailable };
}

/**
 * The message to show for a thrown value.
 *
 * @param error - Whatever was thrown.
 * @param fallbackMessage - What to show when it carries no message.
 * @returns The message.
 */
function synthesisErrorMessage(error: unknown, fallbackMessage: string): string {
  return error instanceof Error && error.message.length > 0
    ? error.message
    : fallbackMessage;
}

/**
 * The operator view, refreshed after a run changed it.
 *
 * @param operatorStatus - The refreshed status.
 * @param jobList - The refreshed job list.
 * @returns The patch.
 */
function synthesisOperatorRefreshState(
  operatorStatus: StudioOperatorStatus,
  jobList: StudioJobListResponse,
): SynthesisOperatorRefreshPatch {
  return {
    auditStatus: operatorStatus.audit,
    jobRecords: jobList.jobs,
    jobStatus: operatorStatus.jobs,
    operatorStatus,
  };
}
