// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/**
 * Pure form-value parsers for Studio admin operator panels.
 *
 * Separated from AdminPanelView so identity/evidence form contracts can be
 * tested without mounting the full multi-section React host.
 */

import type { StudioEvidenceBundleRequest } from "./api/client";

/** Split a comma-separated form field into trimmed non-empty tokens. */
export function textList(value: FormDataEntryValue | null): string[] {
  return String(value ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

/** Return trimmed text, or null when empty. */
export function optionalText(value: FormDataEntryValue | null): string | null {
  const text = String(value ?? "").trim();
  return text.length > 0 ? text : null;
}

/**
 * Parse a form integer clamped to ``[minimum, maximum]``.
 *
 * Non-finite values fall back to ``fallback`` before clamping.
 */
export function boundedInteger(
  value: FormDataEntryValue | null,
  fallback: number,
  minimum: number,
  maximum: number,
): number {
  const parsed = Number(value ?? fallback);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.min(Math.max(Math.trunc(parsed), minimum), maximum);
}

/**
 * Parse a JSON form field into zero or more plain objects.
 *
 * Invalid JSON, primitives, and array non-objects yield an empty list
 * (or only the object elements of an array).
 */
export function jsonObjects(value: FormDataEntryValue | null): Record<string, unknown>[] {
  const text = String(value ?? "").trim();
  if (text.length === 0) {
    return [];
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return [];
  }
  if (Array.isArray(parsed)) {
    return parsed.filter(
      (item): item is Record<string, unknown> =>
        typeof item === "object" && item !== null && !Array.isArray(item),
    );
  }
  if (typeof parsed === "object" && parsed !== null) {
    return [parsed as Record<string, unknown>];
  }
  return [];
}

/** Map identity update checkboxes/roles fields into the API patch body. */
export function identityUpdateFromForm(form: FormData): {
  active: boolean;
  expires_at_utc: null;
  roles: string[];
} {
  return {
    active: form.get("active") === "on",
    expires_at_utc: null,
    roles: textList(form.get("roles")),
  };
}

/** Build a Studio evidence-bundle create request from admin form fields. */
export function evidenceBundleRequestFromForm(form: FormData): StudioEvidenceBundleRequest {
  const method = optionalText(form.get("replayMethod"));
  const route = optionalText(form.get("replayRoute"));
  const requestSha256 = optionalText(form.get("requestSha256"));
  const note = optionalText(form.get("operatorNote"));
  const commandReplay: Record<string, unknown> = {};
  if (method !== null) {
    commandReplay.method = method;
  }
  if (route !== null) {
    commandReplay.route = route;
  }
  if (requestSha256 !== null) {
    commandReplay.request_sha256 = requestSha256;
  }
  if (note !== null) {
    commandReplay.note = note;
  }

  return {
    audit_limit: boundedInteger(form.get("auditLimit"), 100, 1, 1000),
    analysis_results: jsonObjects(form.get("analysisResults")),
    command_replay: Object.keys(commandReplay).length > 0 ? commandReplay : null,
    default_flow_attestations: jsonObjects(form.get("defaultFlowAttestations")),
    default_flow_runs: jsonObjects(form.get("defaultFlowRuns")),
    include_audit: form.get("includeAudit") === "on",
    job_ids: textList(form.get("jobIds")),
    model_scan_results: jsonObjects(form.get("modelScanResults")),
    project_name: optionalText(form.get("projectName")),
    simulation_results: jsonObjects(form.get("simulationResults")),
    weight_restore_results: jsonObjects(form.get("weightRestoreResults")),
    weight_restore_attach_results: jsonObjects(form.get("weightRestoreAttachResults")),
  };
}
