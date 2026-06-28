// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio readiness activation state
import type { StudioOperatorStatus } from "./api/client";

export type StudioReadinessSeverity = "ready" | "warning" | "blocked";

export type StudioReadinessItemKey =
  | "audit"
  | "capabilities"
  | "identity"
  | "jobs"
  | "profile"
  | "resources"
  | "routes";

export interface StudioReadinessItem {
  action: string;
  key: StudioReadinessItemKey;
  label: string;
  status: StudioReadinessSeverity;
  value: string;
}

export interface StudioReadinessModel {
  actionLabel: string;
  blockingCount: number;
  headline: string;
  items: StudioReadinessItem[];
  posture: StudioReadinessSeverity;
  readyCount: number;
  subhead: string;
  warningCount: number;
}

const NOT_LOADED_ITEM: StudioReadinessItem = {
  action: "Refresh operator status",
  key: "profile",
  label: "Operator status",
  status: "warning",
  value: "not loaded",
};

/**
 * Convert the operator status contract into promotion-oriented readiness state.
 *
 * The readiness model does not run new checks or introduce authority. It
 * turns the existing operator status payload into stable UI labels that make
 * local development gaps and production blockers explicit on the first screen.
 */
export function buildStudioReadinessModel(
  status: StudioOperatorStatus | null,
): StudioReadinessModel {
  if (status === null) {
    return finalizeReadiness([NOT_LOADED_ITEM]);
  }
  return finalizeReadiness([
    profileItem(status),
    routePolicyItem(status),
    identityItem(status),
    auditItem(status),
    jobsItem(status),
    resourceLimitItem(status),
    capabilitiesItem(status),
  ]);
}

function finalizeReadiness(items: StudioReadinessItem[]): StudioReadinessModel {
  const blockingCount = items.filter((item) => item.status === "blocked").length;
  const warningCount = items.filter((item) => item.status === "warning").length;
  const readyCount = items.filter((item) => item.status === "ready").length;
  const posture: StudioReadinessSeverity = blockingCount > 0
    ? "blocked"
    : warningCount > 0 ? "warning" : "ready";
  const firstAction = items.find((item) => item.status === posture)?.action;
  return {
    actionLabel: firstAction ?? "Rerun operator status",
    blockingCount,
    headline: headlineForPosture(posture),
    items,
    posture,
    readyCount,
    subhead: `${readyCount}/${items.length} checks ready, ${blockingCount} blocked`,
    warningCount,
  };
}

function headlineForPosture(posture: StudioReadinessSeverity): string {
  switch (posture) {
    case "blocked":
      return "Readiness blocked";
    case "warning":
      return "Readiness has warnings";
    case "ready":
      return "Ready for configured profile";
  }
}

function profileItem(status: StudioOperatorStatus): StudioReadinessItem {
  if (status.deployment_profile === "production") {
    return {
      action: "Rerun preflight",
      key: "profile",
      label: "Profile",
      status: "ready",
      value: "production",
    };
  }
  return {
    action: "Generate deployment profile",
    key: "profile",
    label: "Profile",
    status: "warning",
    value: "development",
  };
}

function routePolicyItem(status: StudioOperatorStatus): StudioReadinessItem {
  const routes = status.route_policies;
  if (routes.enforced && routes.protected_routes_audited) {
    return {
      action: "Review route inventory",
      key: "routes",
      label: "Route policies",
      status: "ready",
      value: `${routes.protected_count}/${routes.total_count} protected`,
    };
  }
  return {
    action: routes.enforced ? "Audit protected routes" : "Enable route policies",
    key: "routes",
    label: "Route policies",
    status: "blocked",
    value: routes.enforced ? "audit incomplete" : "disabled",
  };
}

function identityItem(status: StudioOperatorStatus): StudioReadinessItem {
  const identity = status.identity;
  if (identity.configured && identity.mode === "service_account" && !identity.header_principal_allowed) {
    return {
      action: "Review service accounts",
      key: "identity",
      label: "Identity",
      status: "ready",
      value: "service_account",
    };
  }
  if (identity.configured && identity.mode === "service_account") {
    return {
      action: "Disable header fallback",
      key: "identity",
      label: "Identity",
      status: "warning",
      value: "service_account + header",
    };
  }
  return {
    action: "Bootstrap admin identity",
    key: "identity",
    label: "Identity",
    status: "blocked",
    value: identity.mode,
  };
}

function auditItem(status: StudioOperatorStatus): StudioReadinessItem {
  const audit = status.audit;
  if (audit.configured && audit.path_configured && audit.healthy && audit.sink_type === "jsonl") {
    return {
      action: "Export audit bundle",
      key: "audit",
      label: "Audit",
      status: "ready",
      value: "jsonl healthy",
    };
  }
  if (audit.configured && audit.path_configured) {
    return {
      action: "Repair audit sink",
      key: "audit",
      label: "Audit",
      status: "warning",
      value: audit.healthy ? audit.sink_type : "unhealthy",
    };
  }
  return {
    action: "Configure audit JSONL",
    key: "audit",
    label: "Audit",
    status: "blocked",
    value: audit.sink_type,
  };
}

function jobsItem(status: StudioOperatorStatus): StudioReadinessItem {
  const jobs = status.jobs;
  if (jobs.configured) {
    return {
      action: jobs.timed_out_count > 0 ? "Review timed-out jobs" : "Review job ledger",
      key: "jobs",
      label: "Jobs",
      status: jobs.timed_out_count > 0 ? "warning" : "ready",
      value: `${jobs.completed_count} complete / ${jobs.active_count} active`,
    };
  }
  return {
    action: "Set persistent job root",
    key: "jobs",
    label: "Jobs",
    status: "blocked",
    value: "unconfigured",
  };
}

function resourceLimitItem(status: StudioOperatorStatus): StudioReadinessItem {
  const limits = status.resource_limits;
  const hasJobLimits = limits.job_default_timeout_seconds > 0 && limits.job_max_artifact_bytes > 0;
  const hasEdaLimits = limits.eda_process_cpu_seconds !== null && limits.eda_process_memory_bytes !== null;
  if (hasJobLimits && hasEdaLimits && limits.eda_process_limits_supported) {
    return {
      action: "Review runtime ceilings",
      key: "resources",
      label: "Runtime limits",
      status: "ready",
      value: `${limits.job_default_timeout_seconds}s jobs`,
    };
  }
  if (hasJobLimits && hasEdaLimits) {
    return {
      action: "Confirm host limit support",
      key: "resources",
      label: "Runtime limits",
      status: "warning",
      value: "host unsupported",
    };
  }
  return {
    action: "Set runtime ceilings",
    key: "resources",
    label: "Runtime limits",
    status: "blocked",
    value: "incomplete",
  };
}

function capabilitiesItem(status: StudioOperatorStatus): StudioReadinessItem {
  const capabilities = status.capabilities;
  const value = `${capabilities.healthy_count}/${capabilities.total_count} healthy`;
  if (capabilities.unavailable_count > 0) {
    return {
      action: "Resolve unavailable capability",
      key: "capabilities",
      label: "Capabilities",
      status: "warning",
      value,
    };
  }
  if (capabilities.degraded_count > 0) {
    return {
      action: "Review degraded capability",
      key: "capabilities",
      label: "Capabilities",
      status: "warning",
      value,
    };
  }
  return {
    action: "Review capability registry",
    key: "capabilities",
    label: "Capabilities",
    status: "ready",
    value,
  };
}
