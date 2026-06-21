// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio synthesis store state helpers
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
import { latestSynthesisJobIdWithArtefact } from "./evidenceBundles";

export interface SynthesisOperatorRefreshPatch {
  auditStatus: StudioAuditStatus;
  jobRecords: StudioJobRecord[];
  jobStatus: StudioJobStatus;
  operatorStatus: StudioOperatorStatus;
}

export interface SynthesisRunStartStatePatch {
  activeTab: "synth";
  error: null;
  isSimulating: true;
  latestSynthesisJobId: null;
  multiTargetResult: null;
  synthesisEvidenceBundle: null;
  synthesisEvidenceBundleError: null;
}

export interface MultiTargetSynthesisRunStartStatePatch {
  activeTab: "synth";
  error: null;
  isSimulating: true;
  latestMultiTargetSynthesisJobId: null;
  synthResult: null;
  synthesisEvidenceBundle: null;
  synthesisEvidenceBundleError: null;
}

export interface SynthesisRunCompletedStatePatch
  extends SynthesisOperatorRefreshPatch {
  isSimulating: false;
  latestSynthesisJobId: string | null;
  synthResult: SynthResult;
}

export interface MultiTargetSynthesisRunCompletedStatePatch
  extends SynthesisOperatorRefreshPatch {
  isSimulating: false;
  latestMultiTargetSynthesisJobId: string | null;
  multiTargetResult: MultiTargetResult;
}

export interface SynthesisFailureStatePatch {
  error: string;
  isSimulating: false;
}

export interface SynthesisErrorStatePatch {
  error: string;
}

export interface SynthesisEstimateLoadedStatePatch {
  synthEstimate: SynthEstimate;
}

export interface SynthesisTargetStatePatch {
  synthTarget: string;
}

export interface SynthesisToolStatusLoadedStatePatch {
  toolsAvailable: Record<string, SynthToolInfo>;
}

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

export function synthesisRunCompletedState(
  synthResult: SynthResult,
  operatorStatus: StudioOperatorStatus,
  jobList: StudioJobListResponse,
): SynthesisRunCompletedStatePatch {
  return {
    ...synthesisOperatorRefreshState(operatorStatus, jobList),
    isSimulating: false,
    latestSynthesisJobId: latestSynthesisJobIdWithArtefact(
      jobList.jobs,
      "synthesis/result.json",
    ),
    synthResult,
  };
}

export function multiTargetSynthesisRunCompletedState(
  multiTargetResult: MultiTargetResult,
  operatorStatus: StudioOperatorStatus,
  jobList: StudioJobListResponse,
): MultiTargetSynthesisRunCompletedStatePatch {
  return {
    ...synthesisOperatorRefreshState(operatorStatus, jobList),
    isSimulating: false,
    latestMultiTargetSynthesisJobId: latestSynthesisJobIdWithArtefact(
      jobList.jobs,
      "synthesis/multi-target-result.json",
    ),
    multiTargetResult,
  };
}

export function synthesisFailureState(error: unknown): SynthesisFailureStatePatch {
  return {
    error: synthesisErrorMessage(error, "Synthesis failed"),
    isSimulating: false,
  };
}

export function synthesisErrorState(
  error: unknown,
  fallbackMessage: string,
): SynthesisErrorStatePatch {
  return { error: synthesisErrorMessage(error, fallbackMessage) };
}

export function synthesisErrorMessageState(message: string): SynthesisErrorStatePatch {
  return { error: message };
}

export function synthesisEstimateLoadedState(
  synthEstimate: SynthEstimate,
): SynthesisEstimateLoadedStatePatch {
  return { synthEstimate };
}

export function synthesisTargetState(synthTarget: string): SynthesisTargetStatePatch {
  return { synthTarget };
}

export function synthesisToolStatusLoadedState(
  toolsAvailable: Record<string, SynthToolInfo>,
): SynthesisToolStatusLoadedStatePatch {
  return { toolsAvailable };
}

function synthesisErrorMessage(error: unknown, fallbackMessage: string): string {
  return error instanceof Error && error.message.length > 0
    ? error.message
    : fallbackMessage;
}

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
