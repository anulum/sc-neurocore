import { describe, expect, it } from "vitest";

import type { StudioAuditExport } from "./api/client";
import { summarizeAuditExport } from "./auditShell";

describe("audit shell contract", () => {
  it("summarizes exported audit decisions without exposing paths", () => {
    const exportPayload: StudioAuditExport = {
      configured: true,
      event_count: 6,
      events: [
        {
          action: "studio.simulation.run",
          decision: "allow",
          principal_id: "operator-1",
          reason: "authorized",
          route: "/api/simulate",
          schema_version: "studio.audit.v1",
          request_id: "req-1",
          timestamp_utc: "2026-06-19T18:00:00Z",
          previous_event_hash: null,
          event_hash: "hash-1",
        },
        {
          action: "studio.synth.run",
          decision: "deny",
          principal_id: "operator-2",
          reason: "missing_admin_role",
          route: "/api/synth/run",
          schema_version: "studio.audit.v1",
          request_id: "req-2",
          timestamp_utc: "2026-06-19T18:01:00Z",
          previous_event_hash: "hash-1",
          event_hash: "hash-2",
        },
        {
          action: "studio.audit.export",
          decision: "allow",
          principal_id: "admin-1",
          reason: "authorized",
          route: "/api/studio/audit/export",
          schema_version: "studio.audit.v1",
          request_id: null,
          timestamp_utc: "2026-06-19T18:02:00Z",
          previous_event_hash: "hash-2",
          event_hash: "hash-3",
        },
        {
          action: "studio.auth.login",
          decision: "deny",
          principal_id: null,
          reason: "invalid_browser_login",
          route: "/api/studio/auth/login",
          schema_version: "studio.audit.v1",
          request_id: "req-auth",
          timestamp_utc: "2026-06-19T18:02:30Z",
          previous_event_hash: "hash-3",
          event_hash: "hash-auth",
        },
        {
          action: "studio.identity.browser_user.update",
          decision: "allow",
          principal_id: "admin-1",
          reason: "authorized",
          route: "/api/studio/identity/browser-users/operator",
          schema_version: "studio.audit.v1",
          request_id: "req-3",
          timestamp_utc: "2026-06-19T18:03:00Z",
          previous_event_hash: "hash-auth",
          event_hash: "hash-4",
        },
        {
          action: "studio.identity.service_account.update",
          decision: "deny",
          principal_id: "admin-2",
          reason: "last_admin_guard",
          route: "/api/studio/identity/service-accounts/svc-admin",
          schema_version: "studio.audit.v1",
          request_id: "req-4",
          timestamp_utc: "2026-06-19T18:04:00Z",
          previous_event_hash: "hash-4",
          event_hash: "hash-5",
        },
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
});
