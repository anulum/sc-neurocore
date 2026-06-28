// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio readiness activation state tests
import { describe, expect, it } from "vitest";

import type { StudioOperatorStatus } from "./api/client";
import { buildStudioReadinessModel } from "./studioReadiness";

function operatorStatus(overrides: Partial<StudioOperatorStatus> = {}): StudioOperatorStatus {
  const base: StudioOperatorStatus = {
    audit: {
      configured: true,
      healthy: true,
      last_error: null,
      path_configured: true,
      sink_type: "jsonl",
    },
    browser_login: {
      active_bucket_count: 0,
      cooldown_seconds: 900,
      failure_window_seconds: 300,
      locked_bucket_count: 0,
      max_failures: 5,
      max_retry_after_seconds: 0,
    },
    capabilities: {
      degraded_count: 0,
      experimental_count: 2,
      healthy_count: 10,
      stable_count: 8,
      total_count: 10,
      unavailable_count: 0,
    },
    deployment_profile: "production",
    identity: {
      configured: true,
      header_principal_allowed: false,
      mode: "service_account",
    },
    jobs: {
      active_count: 0,
      allowed_kinds: ["compiler", "evidence", "synthesis", "training"],
      completed_count: 2,
      configured: true,
      failed_count: 0,
      process_count: 2,
      resource_profiles: [],
      schema_version: "studio.jobs.status.v1",
      thread_count: 1,
      timed_out_count: 0,
    },
    resource_limits: {
      eda_process_cpu_seconds: 120,
      eda_process_limits_supported: true,
      eda_process_memory_bytes: 2147483648,
      job_default_timeout_seconds: 300,
      job_max_artifact_bytes: 16777216,
    },
    route_policies: {
      admin_count: 10,
      authenticated_count: 20,
      enforced: true,
      protected_audit_action_count: 30,
      protected_count: 30,
      protected_routes_audited: true,
      public_count: 5,
      total_count: 35,
    },
    schema_version: "studio.operator.status.v1",
  };
  return {
    ...base,
    ...overrides,
  };
}

describe("studio readiness model", () => {
  it("marks a fully configured production operator profile as ready", () => {
    const model = buildStudioReadinessModel(operatorStatus());

    expect(model.posture).toBe("ready");
    expect(model.blockingCount).toBe(0);
    expect(model.warningCount).toBe(0);
    expect(model.readyCount).toBe(7);
    expect(model.headline).toBe("Ready for configured profile");
    expect(model.items.map((item) => item.status)).toEqual([
      "ready",
      "ready",
      "ready",
      "ready",
      "ready",
      "ready",
      "ready",
    ]);
  });

  it("turns local development gaps into explicit blockers and warnings", () => {
    const model = buildStudioReadinessModel(operatorStatus({
      audit: {
        configured: false,
        healthy: true,
        last_error: null,
        path_configured: false,
        sink_type: "memory",
      },
      capabilities: {
        degraded_count: 0,
        experimental_count: 2,
        healthy_count: 9,
        stable_count: 8,
        total_count: 10,
        unavailable_count: 1,
      },
      deployment_profile: "development",
      identity: {
        configured: false,
        header_principal_allowed: true,
        mode: "header_principal",
      },
      jobs: {
        ...operatorStatus().jobs,
        configured: false,
      },
      route_policies: {
        ...operatorStatus().route_policies,
        enforced: false,
      },
    }));

    expect(model.posture).toBe("blocked");
    expect(model.blockingCount).toBe(4);
    expect(model.warningCount).toBe(2);
    expect(model.actionLabel).toBe("Enable route policies");
    expect(model.items.find((item) => item.key === "profile")).toMatchObject({
      action: "Generate deployment profile",
      status: "warning",
      value: "development",
    });
    expect(model.items.find((item) => item.key === "audit")).toMatchObject({
      action: "Configure audit JSONL",
      status: "blocked",
      value: "memory",
    });
    expect(model.items.find((item) => item.key === "capabilities")).toMatchObject({
      action: "Resolve unavailable capability",
      status: "warning",
      value: "9/10 healthy",
    });
  });

  it("handles unloaded operator status without inventing backend checks", () => {
    const model = buildStudioReadinessModel(null);

    expect(model.posture).toBe("warning");
    expect(model.items).toEqual([
      {
        action: "Refresh operator status",
        key: "profile",
        label: "Operator status",
        status: "warning",
        value: "not loaded",
      },
    ]);
  });
});
