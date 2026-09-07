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

/**
 * Read a form entry as text.
 *
 * A `FormData` entry is a string *or* a `File`, and `String(file)` is
 * `"[object File]"` — a value that passes every length check and means
 * nothing. A file submitted where text was expected is a defect in the form,
 * and reading it as empty lets the caller's own validation refuse it instead
 * of storing that string.
 *
 * @param value - The entry, or `null` when the field was absent.
 * @returns The text, or an empty string when there is none.
 */
export function formText(value: FormDataEntryValue | null): string {
  return typeof value === "string" ? value : "";
}

/**
 * Read a comma-separated field as a list.
 *
 * Empty tokens are dropped, so a trailing comma or a doubled one is a typo
 * rather than an empty entry the server has to refuse.
 *
 * @param value - The entry, or `null` when the field was absent.
 * @returns The tokens, trimmed, in the order they were written.
 */
export function textList(value: FormDataEntryValue | null): string[] {
  return formText(value)
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

/**
 * Read an optional field.
 *
 * @param value - The entry, or `null` when the field was absent.
 * @returns The trimmed text, or `null` when the field was blank. Blank and
 *   absent are the same thing here: an untouched input submits as an empty
 *   string, not as nothing.
 */
export function optionalText(value: FormDataEntryValue | null): string | null {
  const text = formText(value).trim();
  return text.length > 0 ? text : null;
}

/**
 * Read a whole number, held inside its bounds.
 *
 * The fallback is used for anything that is not a finite number -- a blank
 * field, text, an infinity -- and is then clamped like any other value, so a
 * fallback outside the bounds cannot smuggle one out.
 *
 * @param value - The entry, or `null` when the field was absent.
 * @param fallback - What to use when the entry is not a number.
 * @param minimum - The lowest value the server accepts.
 * @param maximum - The highest value the server accepts.
 * @returns The bounded whole number.
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
 * Whether a value is a plain object rather than an array or a primitive.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Read a JSON field as a list of plain objects.
 *
 * These fields carry evidence records typed by the server, not by the browser,
 * so anything that is not an object is dropped rather than passed on: invalid
 * JSON, a primitive, an array's non-object elements. A field the operator left
 * alone and a field they filled with nonsense both submit as no records, which
 * is the safe reading -- the alternative is sending the server something it
 * will refuse with an error about a field the operator never touched.
 *
 * @param value - The entry, or `null` when the field was absent.
 * @returns The objects, in the order they were written.
 */
export function jsonObjects(value: FormDataEntryValue | null): Record<string, unknown>[] {
  const text = formText(value).trim();
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
    return parsed.filter(isPlainObject);
  }
  return isPlainObject(parsed) ? [parsed] : [];
}

/**
 * Read the identity form as a patch for the accounts route.
 *
 * `expires_at_utc` is always `null`: this form does not offer an expiry, and
 * sending the field as null is how the route is told not to change it.
 *
 * @param form - The submitted form.
 * @returns The patch body.
 */
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

/**
 * Read the evidence-bundle form as a create request.
 *
 * The replay block is assembled from four optional fields and sent as `null`
 * when none of them was filled in, rather than as an object of nulls. An empty
 * replay block would claim the bundle records how to reproduce itself when it
 * records nothing of the sort.
 *
 * @param form - The submitted form.
 * @returns The request body.
 */
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
