// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio readiness activation state

/**
 * The first screen's answer to "is this deployment ready?"
 *
 * Nothing here runs a check or claims authority it does not have. It reads the
 * operator status the server already publishes and turns it into seven items
 * an operator can act on, each with the action that would improve it.
 *
 * The severities are the point. `blocked` means something must change before
 * this deployment is used in earnest; `warning` means something is worth
 * knowing and is normal in local development. The posture is the worst item,
 * and an absent operator status is a `warning` rather than `ready` -- a
 * deployment that cannot answer is not a deployment that is fine.
 */
import type { StudioOperatorStatus } from "./api/client";

/**
 * How badly an item needs attention. `blocked` must change before the
 * deployment is used in earnest; `warning` is worth knowing and is normal in
 * local development.
 */
export type StudioReadinessSeverity = "ready" | "warning" | "blocked";

/** The seven things readiness looks at. */
export type StudioReadinessItemKey =
  | "audit"
  | "capabilities"
  | "identity"
  | "jobs"
  | "profile"
  | "resources"
  | "routes";

/** One check: what it found, how bad it is, and what to do about it. */
export interface StudioReadinessItem {
  action: string;
  key: StudioReadinessItemKey;
  label: string;
  status: StudioReadinessSeverity;
  value: string;
}

/** Every check, counted and summarised, with the action to take first. */
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

/**
 * What readiness reports when the operator status has not arrived: a warning,
 * not a pass. A deployment that cannot answer is not a deployment that is fine.
 */
const NOT_LOADED_ITEM: StudioReadinessItem = {
  action: "Refresh operator status",
  key: "profile",
  label: "Operator status",
  status: "warning",
  value: "not loaded",
};

/**
 * Read a deployment's readiness off its operator status.
 *
 * @param status - The status, or `null` when it has not arrived.
 * @returns The model: seven items, their counts, the worst posture among them,
 *   and the one action worth taking first. A status that has not arrived
 *   produces a single warning item rather than an empty ready model.
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

/**
 * Count the items and summarise them.
 *
 * The action offered is the first item at the worst severity, so an
 * operator is pointed at something that would actually move the posture
 * rather than at whichever check happens to be first.
 *
 * @param items - The checks, in the order they are shown.
 * @returns The whole model.
 */
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

/**
 * Name a posture in one line.
 *
 * @param posture - The worst severity present.
 * @returns The headline.
 */
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

/**
 * Judge which deployment profile is in force.
 *
 * @param status - The operator status.
 * @returns The readiness item: its value, its severity, and the
 *   action that would improve it.
 */
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

/**
 * Judge whether the protected routes are enforced and audited.
 *
 * @param status - The operator status.
 * @returns The readiness item: its value, its severity, and the
 *   action that would improve it.
 */
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

/**
 * Judge how identities are established.
 *
 * @param status - The operator status.
 * @returns The readiness item: its value, its severity, and the
 *   action that would improve it.
 */
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

/**
 * Judge whether the audit sink is configured and healthy.
 *
 * @param status - The operator status.
 * @returns The readiness item: its value, its severity, and the
 *   action that would improve it.
 */
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

/**
 * Judge whether the job queue is configured and coping.
 *
 * @param status - The operator status.
 * @returns The readiness item: its value, its severity, and the
 *   action that would improve it.
 */
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

/**
 * Judge whether the process limits are in force.
 *
 * @param status - The operator status.
 * @returns The readiness item: its value, its severity, and the
 *   action that would improve it.
 */
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

/**
 * Judge how many registered capabilities are unavailable.
 *
 * @param status - The operator status.
 * @returns The readiness item: its value, its severity, and the
 *   action that would improve it.
 */
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
