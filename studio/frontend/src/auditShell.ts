// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/**
 * The audit trail's summary, and the store patches that move it.
 *
 * Two categories of audit action are counted separately from the rest --
 * browser authentication and identity lifecycle -- because they are the ones an
 * operator is asked about. Each is counted whole, allowed, and denied, and the
 * denied count is the one the headline carries: an audit log with no denials
 * and one with none *recorded* look the same in a total, and the split says
 * which it is.
 *
 * Every patch clears `auditLoading` and sets `auditError` to `null` or to a
 * message, never leaving either as it was. The archive operations go further
 * and carry a whole operator refresh: writing, restoring or purging an archive
 * changes the job list and the audit status too, so those patches replace them
 * from the same snapshot rather than leaving a stale count on screen beside a
 * fresh one.
 */

import type {
  StudioAuditExport,
  StudioAuditQuarantineArchivePurgeResult,
  StudioAuditQuarantineArchiveResult,
  StudioAuditQuarantineArchiveRestoreResult,
  StudioAuditQuarantineArchiveRetentionPlan,
  StudioAuditQuarantineArchiveValidation,
  StudioAuditStatus,
  StudioJobListResponse,
  StudioJobRecord,
  StudioJobStatus,
  StudioOperatorStatus,
} from "./api/client";

/**
 * What the panel says about an audit export: totals, the two categories
 * counted apart, the latest action in each, and a headline.
 */
export interface AuditExportSummary {
  total: number;
  allowed: number;
  denied: number;
  browserAuth: number;
  browserAuthAllowed: number;
  browserAuthDenied: number;
  identityLifecycle: number;
  identityLifecycleAllowed: number;
  identityLifecycleDenied: number;
  truncated: boolean;
  sinkType: string;
  latestAction: string | null;
  latestBrowserAuthAction: string | null;
  latestIdentityLifecycleAction: string | null;
  latestTimestamp: string | null;
  headline: string;
}

/** A request has started: clear the error, show the spinner. */
export interface AuditLoadingStatePatch {
  auditError: null;
  auditLoading: true;
}

/** The audit status arrived. */
export interface AuditStatusLoadedStatePatch {
  auditError: null;
  auditLoading: false;
  auditStatus: StudioAuditStatus;
}

/** The audit export arrived. */
export interface AuditExportLoadedStatePatch {
  auditError: null;
  auditExport: StudioAuditExport;
  auditLoading: false;
}

/** A request failed, with the message to show. */
export interface AuditFailureStatePatch {
  auditError: string;
  auditLoading: false;
}

/**
 * An archive was written. The export is replaced alongside the operator
 * refresh, because archiving moves events out of the live log.
 */
export interface AuditArchiveCreatedStatePatch extends OperatorAuditRefreshPatch {
  auditArchive: StudioAuditQuarantineArchiveResult;
  auditExport: StudioAuditExport;
}

/** The retention plan arrived. */
export interface AuditArchiveRetentionLoadedStatePatch {
  auditArchiveRetention: StudioAuditQuarantineArchiveRetentionPlan;
  auditError: null;
  auditLoading: false;
}

/** An archive was validated. */
export interface AuditArchiveValidationLoadedStatePatch {
  auditArchiveValidation: StudioAuditQuarantineArchiveValidation;
  auditError: null;
  auditLoading: false;
}

/**
 * An archive was restored. The validation is cleared deliberately: it
 * described the archive before the restore, and keeping it would show a verdict
 * about a state that no longer holds.
 */
export interface AuditArchiveRestoredStatePatch extends OperatorAuditRefreshPatch {
  auditArchiveRestore: StudioAuditQuarantineArchiveRestoreResult;
  auditArchiveValidation: null;
}

/**
 * Archives were purged. The retention plan is replaced in the same patch
 * because purging is what the plan describes.
 */
export interface AuditArchivePurgedStatePatch extends OperatorAuditRefreshPatch {
  auditArchivePurge: StudioAuditQuarantineArchivePurgeResult;
  auditArchiveRetention: StudioAuditQuarantineArchiveRetentionPlan;
}

/**
 * What every archive operation refreshes: the audit status, the job list and
 * the operator status, all from one snapshot.
 */
interface OperatorAuditRefreshPatch {
  auditError: null;
  auditLoading: false;
  auditStatus: StudioAuditStatus;
  jobRecords: StudioJobRecord[];
  jobStatus: StudioJobStatus;
  operatorStatus: StudioOperatorStatus;
}

/** Every state the audit-status request can leave the store in. */
export type AuditStatusStatePatch =
  | AuditFailureStatePatch
  | AuditLoadingStatePatch
  | AuditStatusLoadedStatePatch;

/** Every state the audit-export request can leave the store in. */
export type AuditExportStatePatch =
  | AuditExportLoadedStatePatch
  | AuditFailureStatePatch
  | AuditLoadingStatePatch;

/** Every state an audit-archive operation can leave the store in. */
export type AuditArchiveStatePatch =
  | AuditArchiveCreatedStatePatch
  | AuditArchivePurgedStatePatch
  | AuditArchiveRestoredStatePatch
  | AuditArchiveRetentionLoadedStatePatch
  | AuditArchiveValidationLoadedStatePatch
  | AuditFailureStatePatch
  | AuditLoadingStatePatch;

/**
 * Summarise an audit export for the operator panel.
 *
 * An export that never arrived summarises as zeros with a `sinkType` of
 * `unavailable` -- not as an empty log. A deployment with no audit sink and one
 * with an empty one are different states and the panel says which.
 *
 * The totals come from the export's own `event_count` while the breakdowns
 * count the events in hand. They differ when the export is truncated, and the
 * `truncated` flag is carried through so the panel can say so rather than
 * quietly showing a smaller number.
 *
 * @param exportPayload - The export, or `null` when none arrived.
 * @returns The summary.
 */
export function summarizeAuditExport(
  exportPayload: StudioAuditExport | null,
): AuditExportSummary {
  if (exportPayload === null) {
    return {
      total: 0,
      allowed: 0,
      denied: 0,
      browserAuth: 0,
      browserAuthAllowed: 0,
      browserAuthDenied: 0,
      identityLifecycle: 0,
      identityLifecycleAllowed: 0,
      identityLifecycleDenied: 0,
      truncated: false,
      sinkType: "unavailable",
      latestAction: null,
      latestBrowserAuthAction: null,
      latestIdentityLifecycleAction: null,
      latestTimestamp: null,
      headline: "audit export unavailable",
    };
  }
  const allowed = exportPayload.events.filter((event) => event.decision === "allow").length;
  const denied = exportPayload.events.filter((event) => event.decision === "deny").length;
  const browserAuthEvents = exportPayload.events.filter(isBrowserAuthAction);
  const browserAuthAllowed = browserAuthEvents.filter(
    (event) => event.decision === "allow",
  ).length;
  const browserAuthDenied = browserAuthEvents.filter((event) => event.decision === "deny").length;
  const identityLifecycleEvents = exportPayload.events.filter(isIdentityLifecycleAction);
  const identityLifecycleAllowed = identityLifecycleEvents.filter(
    (event) => event.decision === "allow",
  ).length;
  const identityLifecycleDenied = identityLifecycleEvents.filter(
    (event) => event.decision === "deny",
  ).length;
  const latest =
    exportPayload.events.length > 0
      ? exportPayload.events[exportPayload.events.length - 1]
      : null;
  const latestIdentityLifecycle =
    identityLifecycleEvents.length > 0
      ? identityLifecycleEvents[identityLifecycleEvents.length - 1]
      : null;
  const latestBrowserAuth =
    browserAuthEvents.length > 0 ? browserAuthEvents[browserAuthEvents.length - 1] : null;

  return {
    total: exportPayload.event_count,
    allowed,
    denied,
    browserAuth: browserAuthEvents.length,
    browserAuthAllowed,
    browserAuthDenied,
    identityLifecycle: identityLifecycleEvents.length,
    identityLifecycleAllowed,
    identityLifecycleDenied,
    truncated: exportPayload.truncated,
    sinkType: exportPayload.sink_type,
    latestAction: latest?.action ?? null,
    latestBrowserAuthAction: latestBrowserAuth?.action ?? null,
    latestIdentityLifecycleAction: latestIdentityLifecycle?.action ?? null,
    latestTimestamp: latest?.timestamp_utc ?? null,
    headline: `${exportPayload.event_count} events, ${denied} denied`,
  };
}

/**
 * Start an audit request.
 *
 * @returns The patch.
 */
export function auditLoadingState(): AuditLoadingStatePatch {
  return {
    auditError: null,
    auditLoading: true,
  };
}

/**
 * Record an audit status that arrived.
 *
 * @param auditStatus - The status.
 * @returns The patch.
 */
export function auditStatusLoadedState(
  auditStatus: StudioAuditStatus,
): AuditStatusLoadedStatePatch {
  return {
    auditError: null,
    auditLoading: false,
    auditStatus,
  };
}

/**
 * Record an audit export that arrived.
 *
 * @param auditExport - The export.
 * @returns The patch.
 */
export function auditExportLoadedState(
  auditExport: StudioAuditExport,
): AuditExportLoadedStatePatch {
  return {
    auditError: null,
    auditExport,
    auditLoading: false,
  };
}

/**
 * Record an audit request that failed.
 *
 * @param error - What was thrown, which need not be an `Error`.
 * @param fallbackMessage - What to show when it carries no message.
 * @returns The patch.
 */
export function auditFailureState(
  error: unknown,
  fallbackMessage: string,
): AuditFailureStatePatch {
  return {
    auditError: error instanceof Error && error.message.length > 0
      ? error.message
      : fallbackMessage,
    auditLoading: false,
  };
}

/**
 * Record an archive that was written.
 *
 * @param auditArchive - The archive's result.
 * @param auditExport - The audit export as it stands after archiving.
 * @param operatorStatus - The operator status from the same refresh.
 * @param jobList - The job list from the same refresh.
 * @returns The patch.
 */
export function auditArchiveCreatedState(
  auditArchive: StudioAuditQuarantineArchiveResult,
  auditExport: StudioAuditExport,
  operatorStatus: StudioOperatorStatus,
  jobList: StudioJobListResponse,
): AuditArchiveCreatedStatePatch {
  return {
    ...operatorAuditRefreshState(operatorStatus, jobList),
    auditArchive,
    auditExport,
  };
}

/**
 * Record a retention plan that arrived.
 *
 * @param auditArchiveRetention - The plan.
 * @returns The patch.
 */
export function auditArchiveRetentionLoadedState(
  auditArchiveRetention: StudioAuditQuarantineArchiveRetentionPlan,
): AuditArchiveRetentionLoadedStatePatch {
  return {
    auditArchiveRetention,
    auditError: null,
    auditLoading: false,
  };
}

/**
 * Record an archive validation that ran.
 *
 * @param auditArchiveValidation - What it found.
 * @returns The patch.
 */
export function auditArchiveValidationLoadedState(
  auditArchiveValidation: StudioAuditQuarantineArchiveValidation,
): AuditArchiveValidationLoadedStatePatch {
  return {
    auditArchiveValidation,
    auditError: null,
    auditLoading: false,
  };
}

/**
 * Record an archive that was restored.
 *
 * @param auditArchiveRestore - The restore's result.
 * @param operatorStatus - The operator status from the same refresh.
 * @param jobList - The job list from the same refresh.
 * @returns The patch, which also discards the previous validation.
 */
export function auditArchiveRestoredState(
  auditArchiveRestore: StudioAuditQuarantineArchiveRestoreResult,
  operatorStatus: StudioOperatorStatus,
  jobList: StudioJobListResponse,
): AuditArchiveRestoredStatePatch {
  return {
    ...operatorAuditRefreshState(operatorStatus, jobList),
    auditArchiveRestore,
    auditArchiveValidation: null,
  };
}

/**
 * Record archives that were purged.
 *
 * @param auditArchivePurge - The purge's result.
 * @param auditArchiveRetention - The plan as it stands after purging.
 * @param operatorStatus - The operator status from the same refresh.
 * @param jobList - The job list from the same refresh.
 * @returns The patch.
 */
export function auditArchivePurgedState(
  auditArchivePurge: StudioAuditQuarantineArchivePurgeResult,
  auditArchiveRetention: StudioAuditQuarantineArchiveRetentionPlan,
  operatorStatus: StudioOperatorStatus,
  jobList: StudioJobListResponse,
): AuditArchivePurgedStatePatch {
  return {
    ...operatorAuditRefreshState(operatorStatus, jobList),
    auditArchivePurge,
    auditArchiveRetention,
  };
}

/**
 * Whether an audit event is an identity-lifecycle action.
 *
 * @param event - The event.
 * @returns Whether its action is under `studio.identity.`.
 */
function isIdentityLifecycleAction(event: StudioAuditExport["events"][number]): boolean {
  return event.action.startsWith("studio.identity.");
}

/**
 * Whether an audit event is a browser-authentication action.
 *
 * @param event - The event.
 * @returns Whether its action is under `studio.auth.`.
 */
function isBrowserAuthAction(event: StudioAuditExport["events"][number]): boolean {
  return event.action.startsWith("studio.auth.");
}

/**
 * Build the refresh every archive operation carries.
 *
 * @param operatorStatus - The operator status.
 * @param jobList - The job list.
 * @returns The shared part of the patch.
 */
function operatorAuditRefreshState(
  operatorStatus: StudioOperatorStatus,
  jobList: StudioJobListResponse,
): OperatorAuditRefreshPatch {
  return {
    auditError: null,
    auditLoading: false,
    auditStatus: operatorStatus.audit,
    jobRecords: jobList.jobs,
    jobStatus: operatorStatus.jobs,
    operatorStatus,
  };
}
