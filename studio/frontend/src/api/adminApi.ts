// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio frontend API
// Studio API: admin endpoints.
import { post, get, patch, getBlob, encodeArtifactPath } from "./http";
import type {
  StudioCapabilitiesResponse,
  StudioAuditStatus,
  StudioAuditExport,
  StudioAuditQuarantineArchiveResult,
  StudioAuditQuarantineArchiveValidation,
  StudioAuditQuarantineArchiveRestoreResult,
  StudioAuditQuarantineArchiveRetentionPlan,
  StudioAuditQuarantineArchivePurgeResult,
  StudioJobStatus,
  StudioJobRecord,
  StudioJobListResponse,
  StudioEvidenceBundleRequest,
  StudioEvidenceBundleResponse,
  StudioOperatorStatus,
  StudioIdentityServiceAccount,
  StudioIdentityServiceAccountsResponse,
  StudioIdentityBrowserUser,
  StudioIdentityBrowserUsersResponse,
  StudioIdentityServiceAccountUpdate,
  StudioIdentityBrowserUserUpdate,
  StudioIdentityBrowserUserCreate,
  StudioIdentityBrowserUserPasswordRotate,
  StudioAuthSession,
  StudioLoginResponse,
  StudioLogoutResponse,
} from "./types";

/**
 * Ask what this deployment is allowed and able to do.
 *
 * The Studio hides what the server says it cannot do, rather than offering it
 * and failing later, so this is read before the admin surface is drawn.
 *
 * @returns Each capability, whether it is available, and why not if it is not.
 */
export const fetchStudioCapabilities = () =>
  get<StudioCapabilitiesResponse>("/studio/capabilities");

/**
 * Read the audit log's health: its size, its last entry and its quarantine.
 *
 * @returns The audit status as the server holds it.
 */
export const fetchStudioAuditStatus = () =>
  get<StudioAuditStatus>("/studio/audit/status");

/**
 * Export the most recent audit records.
 *
 * @param limit - How many records to take, newest first.
 * @returns The records and the export's own provenance.
 */
export const fetchStudioAuditExport = (limit = 100) =>
  get<StudioAuditExport>(`/studio/audit/export?limit=${encodeURIComponent(limit)}`);

/**
 * Archive quarantined audit records into a signed document.
 *
 * Quarantined records are ones the audit log could not accept as they stood;
 * archiving takes them out of the live log without discarding them.
 *
 * @param limit - How many records to archive.
 * @returns The archive and the manifest that validates it.
 */
export const createStudioAuditQuarantineArchive = (limit = 100) =>
  post<StudioAuditQuarantineArchiveResult>(
    "/studio/audit/quarantine/archive",
    { limit },
  );

/**
 * Check an archive against its manifest before acting on it.
 *
 * The manifest may be absent, which is itself a finding rather than an error:
 * an archive without one cannot be shown to be intact.
 *
 * @param archive - The archive document.
 * @param manifest - Its manifest, or `null` if there is none.
 * @returns What the server could and could not confirm.
 */
export const validateStudioAuditQuarantineArchive = (
  archive: Record<string, unknown>,
  manifest: Record<string, unknown> | null,
) =>
  post<StudioAuditQuarantineArchiveValidation>(
    "/studio/audit/quarantine/archive/validate",
    { archive, manifest },
  );

/**
 * Ask what a purge would remove, without removing it.
 *
 * @param retainLatest - How many archives to keep.
 * @returns The plan: what would be purged and what would be retained.
 */
export const fetchStudioAuditQuarantineArchiveRetention = (retainLatest = 10) =>
  get<StudioAuditQuarantineArchiveRetentionPlan>(
    `/studio/audit/quarantine/archive/retention?retain_latest=${encodeURIComponent(retainLatest)}`,
  );

/**
 * Put an archive's records back into the live audit log.
 *
 * @param archive - The archive document.
 * @param manifest - Its manifest, or `null` if there is none.
 * @returns What was restored and what was refused.
 */
export const restoreStudioAuditQuarantineArchive = (
  archive: Record<string, unknown>,
  manifest: Record<string, unknown> | null,
) =>
  post<StudioAuditQuarantineArchiveRestoreResult>(
    "/studio/audit/quarantine/archive/restore",
    { archive, manifest },
  );

/**
 * Carry out the purge {@link fetchStudioAuditQuarantineArchiveRetention} plans.
 *
 * This is the destructive half, and it is a separate call so the plan can be
 * read first.
 *
 * @param retainLatest - How many archives to keep.
 * @returns What was purged and what was retained.
 */
export const purgeStudioAuditQuarantineArchiveRetention = (retainLatest = 10) =>
  post<StudioAuditQuarantineArchivePurgeResult>(
    "/studio/audit/quarantine/archive/purge",
    { retain_latest: retainLatest },
  );

/**
 * Read the job runner's health and its queue depth.
 *
 * @returns The runner's status.
 */
export const fetchStudioJobStatus = () =>
  get<StudioJobStatus>("/studio/jobs/status");

/**
 * List the jobs the server is holding.
 *
 * @returns One record per job.
 */
export const fetchStudioJobs = () =>
  get<StudioJobListResponse>("/studio/jobs");

/**
 * Read one job by its identifier.
 *
 * @param jobId - The job to read.
 * @returns The job record, artefacts included.
 */
export const fetchStudioJobRecord = (jobId: string) =>
  get<StudioJobRecord>(`/studio/jobs/${encodeURIComponent(jobId)}`);

/**
 * Follow a status route a receipt handed back, whatever prefix it carries.
 *
 * A receipt states where to look rather than how to build the URL, so the
 * route arrives with or without the `/api` prefix and with or without a
 * leading slash. Normalising here means a receipt's own answer is followed
 * instead of a path reassembled from assumptions. An empty route is rejected
 * rather than turned into a request for the job list.
 *
 * @param statusRoute - The route from the receipt.
 * @returns The job record it points at.
 */
export function fetchStudioJobAtStatusRoute(statusRoute: string): Promise<StudioJobRecord> {
  const trimmed = statusRoute.trim();
  if (trimmed.length === 0) {
    return Promise.reject(new Error("empty_status_route"));
  }
  const path = trimmed.startsWith("/api/")
    ? trimmed.slice("/api".length)
    : trimmed.startsWith("/api")
      ? trimmed.slice("/api".length) || "/"
      : trimmed.startsWith("/")
        ? trimmed
        : `/${trimmed}`;
  return get<StudioJobRecord>(path);
}

/**
 * Download one of a job's artefacts.
 *
 * @param jobId - The job that produced it.
 * @param artifactPath - The artefact's path within the job.
 * @returns The artefact's bytes.
 */
export const fetchStudioJobArtifact = (jobId: string, artifactPath: string) =>
  getBlob(
    `/studio/jobs/${encodeURIComponent(jobId)}/artifacts/${encodeArtifactPath(artifactPath)}`,
  );

/**
 * Assemble an evidence bundle from what the server holds.
 *
 * @param request - What to include in the bundle.
 * @returns The bundle and the digests that identify it.
 */
export const createStudioEvidenceBundle = (request: StudioEvidenceBundleRequest) =>
  post<StudioEvidenceBundleResponse>("/studio/evidence/bundle", request);

/**
 * Read the operator view: what is running, what is stale, what needs a person.
 *
 * @returns The operator status.
 */
export const fetchStudioOperatorStatus = () =>
  get<StudioOperatorStatus>("/studio/operator/status");

/**
 * List the service accounts, which are machine principals rather than people.
 *
 * @returns One entry per service account.
 */
export const fetchStudioIdentityServiceAccounts = () =>
  get<StudioIdentityServiceAccountsResponse>("/studio/identity/service-accounts");

/**
 * List the browser users, which are the people who sign in.
 *
 * @returns One entry per user. No credential material is included.
 */
export const fetchStudioIdentityBrowserUsers = () =>
  get<StudioIdentityBrowserUsersResponse>("/studio/identity/browser-users");

/**
 * Create a browser user.
 *
 * @param create - The user to create, with the initial credential.
 * @returns The user as the server created it, without credential material.
 */
export const createStudioIdentityBrowserUser = (
  create: StudioIdentityBrowserUserCreate,
) =>
  post<StudioIdentityBrowserUser>(
    "/studio/identity/browser-users",
    create,
  );

/**
 * Change one service account.
 *
 * A PATCH rather than a PUT, so a concurrent change to a field this caller did
 * not touch is not silently reverted.
 *
 * @param principalId - The account to change.
 * @param update - The fields to change.
 * @returns The account after the change.
 */
export const updateStudioIdentityServiceAccount = (
  principalId: string,
  update: StudioIdentityServiceAccountUpdate,
) =>
  patch<StudioIdentityServiceAccount>(
    `/studio/identity/service-accounts/${encodeURIComponent(principalId)}`,
    update,
  );

/**
 * Change one browser user.
 *
 * @param username - The user to change.
 * @param update - The fields to change.
 * @returns The user after the change.
 */
export const updateStudioIdentityBrowserUser = (
  username: string,
  update: StudioIdentityBrowserUserUpdate,
) =>
  patch<StudioIdentityBrowserUser>(
    `/studio/identity/browser-users/${encodeURIComponent(username)}`,
    update,
  );

/**
 * Rotate a browser user's password.
 *
 * Its own route rather than a field on the update, so changing a credential is
 * never a side effect of editing something else.
 *
 * @param username - The user whose password to rotate.
 * @param update - The rotation request.
 * @returns The user after the rotation, without credential material.
 */
export const rotateStudioIdentityBrowserUserPassword = (
  username: string,
  update: StudioIdentityBrowserUserPasswordRotate,
) =>
  post<StudioIdentityBrowserUser>(
    `/studio/identity/browser-users/${encodeURIComponent(username)}/password`,
    update,
  );

/**
 * Sign a browser user in.
 *
 * The token is returned rather than stored here; `setStudioAuthToken` is what
 * puts it on later requests, so a caller decides when a session begins.
 *
 * @param username - The user signing in.
 * @param password - Their password.
 * @returns The session and its token.
 */
export const loginStudioBrowserUser = (username: string, password: string) =>
  post<StudioLoginResponse>("/studio/auth/login", { username, password });

/**
 * Ask who the current token signs in as, and what it may do.
 *
 * @returns The session the server recognises for the current token.
 */
export const fetchStudioAuthSession = () =>
  get<StudioAuthSession>("/studio/auth/session");

/**
 * Sign the current session out on the server.
 *
 * Clearing the local token is a separate step, so a failed logout does not
 * leave the browser believing it is signed out while the server does not.
 *
 * @returns What the server ended.
 */
export const logoutStudioBrowserUser = () =>
  post<StudioLogoutResponse>("/studio/auth/logout", {});
