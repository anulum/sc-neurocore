import type { StudioAuditExport, StudioAuditStatus } from "./api/client";

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

export interface AuditLoadingStatePatch {
  auditError: null;
  auditLoading: true;
}

export interface AuditStatusLoadedStatePatch {
  auditError: null;
  auditLoading: false;
  auditStatus: StudioAuditStatus;
}

export interface AuditExportLoadedStatePatch {
  auditError: null;
  auditExport: StudioAuditExport;
  auditLoading: false;
}

export interface AuditFailureStatePatch {
  auditError: string;
  auditLoading: false;
}

export type AuditStatusStatePatch =
  | AuditFailureStatePatch
  | AuditLoadingStatePatch
  | AuditStatusLoadedStatePatch;

export type AuditExportStatePatch =
  | AuditExportLoadedStatePatch
  | AuditFailureStatePatch
  | AuditLoadingStatePatch;

/** Derive operator-facing audit export statistics from the backend payload. */
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

export function auditLoadingState(): AuditLoadingStatePatch {
  return {
    auditError: null,
    auditLoading: true,
  };
}

export function auditStatusLoadedState(
  auditStatus: StudioAuditStatus,
): AuditStatusLoadedStatePatch {
  return {
    auditError: null,
    auditLoading: false,
    auditStatus,
  };
}

export function auditExportLoadedState(
  auditExport: StudioAuditExport,
): AuditExportLoadedStatePatch {
  return {
    auditError: null,
    auditExport,
    auditLoading: false,
  };
}

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

function isIdentityLifecycleAction(event: StudioAuditExport["events"][number]): boolean {
  return event.action.startsWith("studio.identity.");
}

function isBrowserAuthAction(event: StudioAuditExport["events"][number]): boolean {
  return event.action.startsWith("studio.auth.");
}
