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

export const fetchStudioCapabilities = () =>
  get<StudioCapabilitiesResponse>("/studio/capabilities");

export const fetchStudioAuditStatus = () =>
  get<StudioAuditStatus>("/studio/audit/status");

export const fetchStudioAuditExport = (limit = 100) =>
  get<StudioAuditExport>(`/studio/audit/export?limit=${encodeURIComponent(limit)}`);

export const createStudioAuditQuarantineArchive = (limit = 100) =>
  post<StudioAuditQuarantineArchiveResult>(
    "/studio/audit/quarantine/archive",
    { limit },
  );

export const validateStudioAuditQuarantineArchive = (
  archive: Record<string, unknown>,
  manifest: Record<string, unknown> | null,
) =>
  post<StudioAuditQuarantineArchiveValidation>(
    "/studio/audit/quarantine/archive/validate",
    { archive, manifest },
  );

export const fetchStudioAuditQuarantineArchiveRetention = (retainLatest = 10) =>
  get<StudioAuditQuarantineArchiveRetentionPlan>(
    `/studio/audit/quarantine/archive/retention?retain_latest=${encodeURIComponent(retainLatest)}`,
  );

export const restoreStudioAuditQuarantineArchive = (
  archive: Record<string, unknown>,
  manifest: Record<string, unknown> | null,
) =>
  post<StudioAuditQuarantineArchiveRestoreResult>(
    "/studio/audit/quarantine/archive/restore",
    { archive, manifest },
  );

export const purgeStudioAuditQuarantineArchiveRetention = (retainLatest = 10) =>
  post<StudioAuditQuarantineArchivePurgeResult>(
    "/studio/audit/quarantine/archive/purge",
    { retain_latest: retainLatest },
  );

export const fetchStudioJobStatus = () =>
  get<StudioJobStatus>("/studio/jobs/status");

export const fetchStudioJobs = () =>
  get<StudioJobListResponse>("/studio/jobs");

export const fetchStudioJobRecord = (jobId: string) =>
  get<StudioJobRecord>(`/studio/jobs/${encodeURIComponent(jobId)}`);

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

export const fetchStudioJobArtifact = (jobId: string, artifactPath: string) =>
  getBlob(
    `/studio/jobs/${encodeURIComponent(jobId)}/artifacts/${encodeArtifactPath(artifactPath)}`,
  );

export const createStudioEvidenceBundle = (request: StudioEvidenceBundleRequest) =>
  post<StudioEvidenceBundleResponse>("/studio/evidence/bundle", request);

export const fetchStudioOperatorStatus = () =>
  get<StudioOperatorStatus>("/studio/operator/status");

export const fetchStudioIdentityServiceAccounts = () =>
  get<StudioIdentityServiceAccountsResponse>("/studio/identity/service-accounts");

export const fetchStudioIdentityBrowserUsers = () =>
  get<StudioIdentityBrowserUsersResponse>("/studio/identity/browser-users");

export const createStudioIdentityBrowserUser = (
  create: StudioIdentityBrowserUserCreate,
) =>
  post<StudioIdentityBrowserUser>(
    "/studio/identity/browser-users",
    create,
  );

export const updateStudioIdentityServiceAccount = (
  principalId: string,
  update: StudioIdentityServiceAccountUpdate,
) =>
  patch<StudioIdentityServiceAccount>(
    `/studio/identity/service-accounts/${encodeURIComponent(principalId)}`,
    update,
  );

export const updateStudioIdentityBrowserUser = (
  username: string,
  update: StudioIdentityBrowserUserUpdate,
) =>
  patch<StudioIdentityBrowserUser>(
    `/studio/identity/browser-users/${encodeURIComponent(username)}`,
    update,
  );

export const rotateStudioIdentityBrowserUserPassword = (
  username: string,
  update: StudioIdentityBrowserUserPasswordRotate,
) =>
  post<StudioIdentityBrowserUser>(
    `/studio/identity/browser-users/${encodeURIComponent(username)}/password`,
    update,
  );

export const loginStudioBrowserUser = (username: string, password: string) =>
  post<StudioLoginResponse>("/studio/auth/login", { username, password });

export const fetchStudioAuthSession = () =>
  get<StudioAuthSession>("/studio/auth/session");

export const logoutStudioBrowserUser = () =>
  post<StudioLogoutResponse>("/studio/auth/logout", {});

