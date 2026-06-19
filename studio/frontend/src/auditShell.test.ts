import { describe, expect, it } from "vitest";

import type { StudioAuditExport } from "./api/client";
import { summarizeAuditExport } from "./auditShell";

describe("audit shell contract", () => {
  it("summarizes exported audit decisions without exposing paths", () => {
    const exportPayload: StudioAuditExport = {
      configured: true,
      event_count: 3,
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
      ],
      schema_version: "studio.audit.export.v1",
      sink_type: "jsonl",
      truncated: false,
    };

    expect(summarizeAuditExport(exportPayload)).toEqual({
      total: 3,
      allowed: 2,
      denied: 1,
      truncated: false,
      sinkType: "jsonl",
      latestAction: "studio.audit.export",
      latestTimestamp: "2026-06-19T18:02:00Z",
      headline: "3 events, 1 denied",
    });
  });

  it("returns an empty summary for unavailable exports", () => {
    expect(summarizeAuditExport(null)).toEqual({
      total: 0,
      allowed: 0,
      denied: 0,
      truncated: false,
      sinkType: "unavailable",
      latestAction: null,
      latestTimestamp: null,
      headline: "audit export unavailable",
    });
  });
});
