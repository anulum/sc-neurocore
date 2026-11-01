// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Actor and workspace scoped graph job recovery

import type { PreparedGraphJob } from "./api/graphJobsApi";
import { graphJobInputKey } from "./graphJobRequest";
import { parseGraphJobValidation } from "./graphJobs";
import { readGraphJobSavedInput } from "./graphJobSavedInput";
import { parseStudioJson, stringifyStudioJson } from "./studioJson";

/** Retained original submission, independently rechecked before an explicit retry. */
export interface GraphJobRetryInput {
  requestJson: string;
  retryKey: string;
  validation: unknown;
}

/** Persisted recovery metadata, never trusted scientific result data or an auth token. */
export interface GraphJobRecovery {
  schema_version: "studio.graph.job-recovery.v1";
  owner: string;
  selectedJobId: string | null;
  submission: GraphJobRetryInput | null;
}

/** The browser-tab pointer lets an unsaved graph's selected job survive a reload. */
const CURRENT_WORKSPACE = "sc-neurocore-current-graph-workspace";
/** A saved workspace identifier is data, never a URL or filesystem path. */
const WORKSPACE_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

/**
 * Read a workspace UUID, assigning a distinct identity only to a legacy absent field.
 *
 * @param value - Saved workspace identity, or undefined for an older document.
 * @returns The original UUID, or a fresh identity for the legacy workspace.
 * @throws {Error} When an explicit identity is malformed.
 */
export function graphWorkspaceId(value?: unknown): string {
  if (value === undefined) return crypto.randomUUID();
  if (typeof value !== "string" || !WORKSPACE_ID.test(value)) throw new Error("Saved graph workspace identity is invalid.");
  return value;
}

/**
 * Recover this tab's last workspace pointer without treating storage access as mandatory startup.
 *
 * @returns The retained UUID, or a fresh workspace when the browser cannot supply it.
 */
export function initialGraphWorkspaceId(): string {
  try { return graphWorkspaceId(sessionStorage.getItem(CURRENT_WORKSPACE) ?? undefined); }
  catch { return graphWorkspaceId(); }
}

/**
 * Read only recovery metadata; a retained validation remains unknown until its seal is checked.
 *
 * @param value - Saved browser or project recovery block.
 * @returns A detached recovery record, or null for an absent legacy block.
 * @throws {Error} When present metadata or retained input is malformed or unbounded.
 */
export function readGraphJobRecovery(value: unknown): GraphJobRecovery | null {
  if (value === undefined || value === null) return null;
  if (typeof value !== "object" || Array.isArray(value) || !("schema_version" in value)
    || value.schema_version !== "studio.graph.job-recovery.v1" || !("owner" in value)
    || typeof value.owner !== "string" || value.owner.length === 0 || !("selectedJobId" in value)
    || value.selectedJobId !== null && (typeof value.selectedJobId !== "string" || !/^sj_[0-9a-f]{16}$/.test(value.selectedJobId))
    || !("submission" in value)) throw new Error("Saved graph job recovery is invalid.");
  const raw = value.submission;
  let submission: GraphJobRetryInput | null = null;
  if (raw !== null) {
    if (typeof raw !== "object" || Array.isArray(raw) || !("requestJson" in raw)
      || typeof raw.requestJson !== "string" || raw.requestJson.length === 0 || new TextEncoder().encode(raw.requestJson).length > 16 * 1024 * 1024
      || !("retryKey" in raw) || typeof raw.retryKey !== "string" || !/^[!-~]{1,128}$/.test(raw.retryKey)
      || !("validation" in raw)) throw new Error("Saved graph job retry input is invalid.");
    submission = { requestJson: raw.requestJson, retryKey: raw.retryKey, validation: structuredClone(raw.validation) };
  }
  return { schema_version: "studio.graph.job-recovery.v1", owner: value.owner, selectedJobId: value.selectedJobId, submission };
}

/**
 * Verify retained original input and validation without allocating or replacing its retry key.
 *
 * @param submission - Retained wire JSON and unknown validation.
 * @returns Exact original bytes and an independently checked preparation.
 */
export async function prepareGraphJobRecovery(submission: GraphJobRetryInput): Promise<PreparedGraphJob> {
  const parsed = parseStudioJson(submission.requestJson);
  const graph = readGraphJobSavedInput(parsed);
  const request = { json: submission.requestJson, inputKey: graphJobInputKey(graph) };
  const identity = await parseGraphJobValidation(submission.validation, parsed);
  return { request, identity };
}

/**
 * Construct one actor-owned browser cache key, never shared with another workspace.
 *
 * @param workspace - Current stable saved workspace UUID.
 * @param owner - Authenticated actor, or the existing development local actor.
 * @returns Its exact browser storage key.
 */
function storageKey(workspace: string, owner: string): string {
  return `sc-neurocore-graph-job:${graphWorkspaceId(workspace)}:${encodeURIComponent(owner)}`;
}

/**
 * Read the scoped recovery cache without using another actor's selected job.
 *
 * @param workspace - Current workspace UUID.
 * @param owner - Current actor.
 * @returns Checked metadata, or null if this actor has no cached recovery.
 */
export function loadGraphJobRecovery(workspace: string, owner: string): GraphJobRecovery | null {
  const raw = localStorage.getItem(storageKey(workspace, owner));
  const recovery = raw === null ? null : readGraphJobRecovery(parseStudioJson(raw));
  if (recovery !== null && recovery.owner !== owner) throw new Error("Saved graph recovery belongs to another account.");
  return recovery;
}

/**
 * Retain retry intent before allocating a calculation; a failed write must stop submission.
 *
 * @param workspace - Current workspace UUID.
 * @param recovery - Exact original request/key or selected actor-owned job pointer.
 * @throws {Error} When the browser refuses either storage write.
 */
export function storeGraphJobRecovery(workspace: string, recovery: GraphJobRecovery): void {
  const key = storageKey(workspace, recovery.owner);
  const text = stringifyStudioJson(readGraphJobRecovery(recovery));
  localStorage.setItem(key, text);
  sessionStorage.setItem(CURRENT_WORKSPACE, workspace);
}
