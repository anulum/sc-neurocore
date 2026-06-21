import { describe, expect, it } from "vitest";

import type { StudioAuditEvent, StudioAuditExport, StudioAuditStatus } from "./api/client";
import {
  auditExportLoadedState,
  auditFailureState,
  auditLoadingState,
  auditStatusLoadedState,
  summarizeAuditExport,
} from "./auditShell";

function auditStatus(overrides: Partial<StudioAuditStatus> = {}): StudioAuditStatus {
  return {
    configured: overrides.configured ?? true,
    healthy: overrides.healthy ?? true,
    last_error: overrides.last_error ?? null,
    path_configured: overrides.path_configured ?? true,
    sink_type: overrides.sink_type ?? "jsonl",
  };
}

function auditEvent(overrides: Partial<StudioAuditEvent> = {}): StudioAuditEvent {
  return {
    action: overrides.action ?? "studio.audit.status",
    decision: overrides.decision ?? "allow",
    event_hash: overrides.event_hash ?? "event-hash",
    previous_event_hash: overrides.previous_event_hash ?? null,
    principal_id: overrides.principal_id ?? "operator",
    reason: overrides.reason ?? "contract test",
    request_id: overrides.request_id ?? "req-1",
    route: overrides.route ?? "/studio/audit/status",
    schema_version: overrides.schema_version ?? "studio.audit.event.v1",
    timestamp_utc: overrides.timestamp_utc ?? "2026-06-21T12:00:00Z",
  };
}

function auditExport(overrides: Partial<StudioAuditExport> = {}): StudioAuditExport {
  return {
    configured: overrides.configured ?? true,
    event_count: overrides.event_count ?? 1,
    events: overrides.events ?? [auditEvent()],
    schema_version: overrides.schema_version ?? "studio.audit.export.v1",
    sink_type: overrides.sink_type ?? "jsonl",
    truncated: overrides.truncated ?? false,
  };
}

describe("audit shell state contract", () => {
  it("summarizes exported audit decisions without exposing paths", () => {
    const exportPayload: StudioAuditExport = {
      configured: true,
      event_count: 6,
      events: [
        auditEvent({
          action: "studio.simulation.run",
          decision: "allow",
          event_hash: "hash-1",
          previous_event_hash: null,
          principal_id: "operator-1",
          reason: "authorized",
          request_id: "req-1",
          route: "/api/simulate",
          schema_version: "studio.audit.v1",
          timestamp_utc: "2026-06-19T18:00:00Z",
        }),
        auditEvent({
          action: "studio.synth.run",
          decision: "deny",
          event_hash: "hash-2",
          previous_event_hash: "hash-1",
          principal_id: "operator-2",
          reason: "missing_admin_role",
          request_id: "req-2",
          route: "/api/synth/run",
          schema_version: "studio.audit.v1",
          timestamp_utc: "2026-06-19T18:01:00Z",
        }),
        auditEvent({
          action: "studio.audit.export",
          decision: "allow",
          event_hash: "hash-3",
          previous_event_hash: "hash-2",
          principal_id: "admin-1",
          reason: "authorized",
          request_id: null,
          route: "/api/studio/audit/export",
          schema_version: "studio.audit.v1",
          timestamp_utc: "2026-06-19T18:02:00Z",
        }),
        auditEvent({
          action: "studio.auth.login",
          decision: "deny",
          event_hash: "hash-auth",
          previous_event_hash: "hash-3",
          principal_id: null,
          reason: "invalid_browser_login",
          request_id: "req-auth",
          route: "/api/studio/auth/login",
          schema_version: "studio.audit.v1",
          timestamp_utc: "2026-06-19T18:02:30Z",
        }),
        auditEvent({
          action: "studio.identity.browser_user.update",
          decision: "allow",
          event_hash: "hash-4",
          previous_event_hash: "hash-auth",
          principal_id: "admin-1",
          reason: "authorized",
          request_id: "req-3",
          route: "/api/studio/identity/browser-users/operator",
          schema_version: "studio.audit.v1",
          timestamp_utc: "2026-06-19T18:03:00Z",
        }),
        auditEvent({
          action: "studio.identity.service_account.update",
          decision: "deny",
          event_hash: "hash-5",
          previous_event_hash: "hash-4",
          principal_id: "admin-2",
          reason: "last_admin_guard",
          request_id: "req-4",
          route: "/api/studio/identity/service-accounts/svc-admin",
          schema_version: "studio.audit.v1",
          timestamp_utc: "2026-06-19T18:04:00Z",
        }),
      ],
      schema_version: "studio.audit.export.v1",
      sink_type: "jsonl",
      truncated: false,
    };

    expect(summarizeAuditExport(exportPayload)).toEqual({
      total: 6,
      allowed: 3,
      denied: 3,
      browserAuth: 1,
      browserAuthAllowed: 0,
      browserAuthDenied: 1,
      identityLifecycle: 2,
      identityLifecycleAllowed: 1,
      identityLifecycleDenied: 1,
      truncated: false,
      sinkType: "jsonl",
      latestAction: "studio.identity.service_account.update",
      latestBrowserAuthAction: "studio.auth.login",
      latestIdentityLifecycleAction: "studio.identity.service_account.update",
      latestTimestamp: "2026-06-19T18:04:00Z",
      headline: "6 events, 3 denied",
    });
  });

  it("returns an empty summary for unavailable exports", () => {
    expect(summarizeAuditExport(null)).toEqual({
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
    });
  });

  it("builds audit loading state patches", () => {
    expect(auditLoadingState()).toEqual({
      auditError: null,
      auditLoading: true,
    });
  });

  it("builds audit status success state patches", () => {
    const status = auditStatus({ healthy: false, last_error: "sink offline" });

    expect(auditStatusLoadedState(status)).toEqual({
      auditError: null,
      auditLoading: false,
      auditStatus: status,
    });
  });

  it("builds audit export success state patches", () => {
    const exported = auditExport({
      events: [auditEvent({ action: "studio.audit.export" })],
    });

    expect(auditExportLoadedState(exported)).toEqual({
      auditError: null,
      auditExport: exported,
      auditLoading: false,
    });
  });

  it("builds audit failure state patches with fallback messages", () => {
    expect(auditFailureState(new Error("audit sink offline"), "Audit failed")).toEqual({
      auditError: "audit sink offline",
      auditLoading: false,
    });
    expect(auditFailureState("bad", "Audit failed")).toEqual({
      auditError: "Audit failed",
      auditLoading: false,
    });
  });
});
