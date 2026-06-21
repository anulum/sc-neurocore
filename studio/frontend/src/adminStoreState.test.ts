// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio admin store state helper tests
import { describe, expect, it } from "vitest";

import type {
  StudioAuditExport,
  StudioAuditStatus,
  StudioIdentityBrowserUsersResponse,
  StudioIdentityServiceAccountsResponse,
  StudioJobListResponse,
  StudioJobStatus,
  StudioOperatorStatus,
} from "./api/client";
import {
  adminBusyState,
  adminFailureState,
  identityAccountsLoadedState,
  identityAccountsMutatedState,
  jobStatusLoadedState,
  operatorStatusLoadedState,
} from "./adminStoreState";

function auditStatus(overrides: Partial<StudioAuditStatus> = {}): StudioAuditStatus {
  return {
    configured: overrides.configured ?? true,
    healthy: overrides.healthy ?? true,
    last_error: overrides.last_error ?? null,
    path_configured: overrides.path_configured ?? true,
    sink_type: overrides.sink_type ?? "jsonl",
  };
}

function auditExport(overrides: Partial<StudioAuditExport> = {}): StudioAuditExport {
  return {
    configured: overrides.configured ?? true,
    event_count: overrides.event_count ?? 1,
    events: overrides.events ?? [],
    schema_version: overrides.schema_version ?? "studio.audit.export.v1",
    sink_type: overrides.sink_type ?? "jsonl",
    truncated: overrides.truncated ?? false,
  };
}

function jobStatus(overrides: Partial<StudioJobStatus> = {}): StudioJobStatus {
  return {
    active_count: overrides.active_count ?? 1,
    allowed_kinds: overrides.allowed_kinds ?? ["evidence_bundle"],
    completed_count: overrides.completed_count ?? 2,
    configured: overrides.configured ?? true,
    failed_count: overrides.failed_count ?? 0,
    process_count: overrides.process_count ?? 0,
    resource_profiles: overrides.resource_profiles ?? [],
    schema_version: overrides.schema_version ?? "studio.jobs.status.v1",
    thread_count: overrides.thread_count ?? 1,
    timed_out_count: overrides.timed_out_count ?? 0,
  };
}

function jobList(overrides: Partial<StudioJobListResponse> = {}): StudioJobListResponse {
  return {
    jobs: overrides.jobs ?? [],
    schema_version: overrides.schema_version ?? "studio.jobs.list.v1",
  };
}

function accountsResponse(
  overrides: Partial<StudioIdentityServiceAccountsResponse> = {},
): StudioIdentityServiceAccountsResponse {
  return {
    schema_version: overrides.schema_version ?? "studio.identity.service-accounts.v1",
    service_accounts: overrides.service_accounts ?? [{
      active: true,
      expires_at_utc: null,
      principal_id: "svc-admin",
      roles: ["admin"],
    }],
  };
}

function usersResponse(
  overrides: Partial<StudioIdentityBrowserUsersResponse> = {},
): StudioIdentityBrowserUsersResponse {
  return {
    browser_users: overrides.browser_users ?? [{
      active: true,
      expires_at_utc: null,
      principal_id: "browser-admin",
      roles: ["admin"],
      username: "operator",
    }],
    schema_version: overrides.schema_version ?? "studio.identity.browser-users.v1",
  };
}

function operatorStatus(overrides: Partial<StudioOperatorStatus> = {}): StudioOperatorStatus {
  return {
    audit: overrides.audit ?? auditStatus(),
    browser_login: overrides.browser_login ?? {
      active_bucket_count: 0,
      cooldown_seconds: 60,
      failure_window_seconds: 300,
      locked_bucket_count: 0,
      max_failures: 5,
      max_retry_after_seconds: 0,
    },
    capabilities: overrides.capabilities ?? {
      degraded_count: 0,
      experimental_count: 0,
      healthy_count: 4,
      stable_count: 4,
      total_count: 4,
      unavailable_count: 0,
    },
    deployment_profile: overrides.deployment_profile ?? "production",
    identity: overrides.identity ?? {
      configured: true,
      header_principal_allowed: false,
      mode: "service_account",
    },
    jobs: overrides.jobs ?? jobStatus(),
    resource_limits: overrides.resource_limits ?? {
      eda_process_cpu_seconds: null,
      eda_process_limits_supported: false,
      eda_process_memory_bytes: null,
      job_default_timeout_seconds: 600,
      job_max_artifact_bytes: 1048576,
    },
    route_policies: overrides.route_policies ?? {
      admin_count: 2,
      authenticated_count: 5,
      enforced: true,
      protected_audit_action_count: 2,
      protected_count: 7,
      protected_routes_audited: true,
      public_count: 3,
      total_count: 10,
    },
    schema_version: overrides.schema_version ?? "studio.operator.status.v1",
  };
}

describe("admin store state helpers", () => {
  it("builds shared busy and failure patches", () => {
    expect(adminBusyState()).toEqual({ auditError: null, auditLoading: true });
    expect(adminFailureState(new Error("jobs offline"), "fallback")).toEqual({
      auditError: "jobs offline",
      auditLoading: false,
    });
    expect(adminFailureState("bad", "fallback")).toEqual({
      auditError: "fallback",
      auditLoading: false,
    });
  });

  it("builds job status patches", () => {
    const status = jobStatus({ completed_count: 4 });
    const jobs = jobList();

    expect(jobStatusLoadedState(status, jobs)).toEqual({
      auditError: null,
      auditLoading: false,
      jobRecords: jobs.jobs,
      jobStatus: status,
    });
  });

  it("builds identity account refresh and mutation patches", () => {
    const accounts = accountsResponse();
    const users = usersResponse();
    const exported = auditExport();

    expect(identityAccountsLoadedState(accounts, users)).toEqual({
      auditError: null,
      auditLoading: false,
      identityBrowserUsers: users.browser_users,
      identityServiceAccounts: accounts.service_accounts,
    });
    expect(identityAccountsMutatedState(accounts, users, exported)).toEqual({
      auditError: null,
      auditExport: exported,
      auditLoading: false,
      identityBrowserUsers: users.browser_users,
      identityServiceAccounts: accounts.service_accounts,
    });
  });

  it("builds operator status patches", () => {
    const operator = operatorStatus({
      audit: auditStatus({ healthy: false, last_error: "sink warning" }),
      jobs: jobStatus({ completed_count: 5 }),
    });
    const jobs = jobList();

    expect(operatorStatusLoadedState(operator, jobs)).toEqual({
      auditError: null,
      auditLoading: false,
      auditStatus: operator.audit,
      jobRecords: jobs.jobs,
      jobStatus: operator.jobs,
      operatorStatus: operator,
    });
  });
});
