import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { StudioAuditExport, StudioAuditStatus, StudioCapability } from "../api/client";
import { buildAdminShellModel } from "../adminShell";
import AdminPanelView from "./AdminPanelView";

function capability(overrides: Partial<StudioCapability> = {}): StudioCapability {
  return {
    capability_id: overrides.capability_id ?? "studio.capability_registry",
    title: overrides.title ?? "Capability Registry",
    summary: overrides.summary ?? "Registry.",
    status: overrides.status ?? "stable",
    healthy: overrides.healthy ?? true,
    message: overrides.message ?? "Ready.",
    requirements: overrides.requirements ?? [],
    evidence: overrides.evidence ?? ["contract_test"],
    ui_placement: overrides.ui_placement ?? "Admin",
    docs_path: overrides.docs_path ?? "docs/studio/index.md",
  };
}

const auditStatus: StudioAuditStatus = {
  configured: true,
  healthy: false,
  last_error: "AuditPathIsDirectory",
  path_configured: true,
  sink_type: "jsonl",
};

const auditExport: StudioAuditExport = {
  configured: true,
  event_count: 1,
  events: [
    {
      action: "studio.audit.export",
      decision: "deny",
      event_hash: "hash-1",
      previous_event_hash: null,
      principal_id: "operator-1",
      reason: "missing_admin_role",
      request_id: "req-1",
      route: "/api/studio/audit/export",
      schema_version: "studio.audit.v1",
      timestamp_utc: "2026-06-19T20:01:00Z",
    },
  ],
  schema_version: "studio.audit.export.v1",
  sink_type: "jsonl",
  truncated: false,
};

describe("AdminPanel", () => {
  it("renders audit health, denied events, and degraded capabilities", () => {
    const model = buildAdminShellModel({
      auditError: "Audit export failed",
      auditExport,
      auditStatus,
      capabilities: [
        capability(),
        capability({
          capability_id: "studio.synthesis_dashboard",
          title: "Synthesis Dashboard",
          status: "unavailable",
          healthy: false,
          message: "Yosys unavailable.",
        }),
      ],
    });

    const html = renderToStaticMarkup(
      <AdminPanelView
        auditLoading={false}
        model={model}
        onLoadAuditExport={async () => undefined}
        onLoadAuditStatus={async () => undefined}
      />,
    );

    expect(html).toContain("Audit");
    expect(html).toContain("jsonl");
    expect(html).toContain("unhealthy");
    expect(html).toContain("1");
    expect(html).toContain("missing_admin_role");
    expect(html).toContain("Synthesis Dashboard");
    expect(html).toContain("Yosys unavailable.");
  });
});
