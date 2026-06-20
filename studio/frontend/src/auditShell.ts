import type { StudioAuditExport } from "./api/client";

export interface AuditExportSummary {
  total: number;
  allowed: number;
  denied: number;
  identityLifecycle: number;
  truncated: boolean;
  sinkType: string;
  latestAction: string | null;
  latestIdentityLifecycleAction: string | null;
  latestTimestamp: string | null;
  headline: string;
}

/** Derive operator-facing audit export statistics from the backend payload. */
export function summarizeAuditExport(
  exportPayload: StudioAuditExport | null,
): AuditExportSummary {
  if (exportPayload === null) {
    return {
      total: 0,
      allowed: 0,
      denied: 0,
      identityLifecycle: 0,
      truncated: false,
      sinkType: "unavailable",
      latestAction: null,
      latestIdentityLifecycleAction: null,
      latestTimestamp: null,
      headline: "audit export unavailable",
    };
  }
  const allowed = exportPayload.events.filter((event) => event.decision === "allow").length;
  const denied = exportPayload.events.filter((event) => event.decision === "deny").length;
  const identityLifecycleEvents = exportPayload.events.filter(isIdentityLifecycleAction);
  const latest =
    exportPayload.events.length > 0
      ? exportPayload.events[exportPayload.events.length - 1]
      : null;
  const latestIdentityLifecycle =
    identityLifecycleEvents.length > 0
      ? identityLifecycleEvents[identityLifecycleEvents.length - 1]
      : null;

  return {
    total: exportPayload.event_count,
    allowed,
    denied,
    identityLifecycle: identityLifecycleEvents.length,
    truncated: exportPayload.truncated,
    sinkType: exportPayload.sink_type,
    latestAction: latest?.action ?? null,
    latestIdentityLifecycleAction: latestIdentityLifecycle?.action ?? null,
    latestTimestamp: latest?.timestamp_utc ?? null,
    headline: `${exportPayload.event_count} events, ${denied} denied`,
  };
}

function isIdentityLifecycleAction(event: StudioAuditExport["events"][number]): boolean {
  return event.action.startsWith("studio.identity.");
}
