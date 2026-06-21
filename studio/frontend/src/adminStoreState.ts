// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio admin store state helpers
import type {
  StudioAuditExport,
  StudioIdentityBrowserUser,
  StudioIdentityBrowserUsersResponse,
  StudioIdentityServiceAccount,
  StudioIdentityServiceAccountsResponse,
  StudioJobListResponse,
  StudioJobRecord,
  StudioJobStatus,
  StudioOperatorStatus,
} from "./api/client";

export interface AdminBusyStatePatch {
  auditError: null;
  auditLoading: true;
}

export interface AdminFailureStatePatch {
  auditError: string;
  auditLoading: false;
}

export interface JobStatusLoadedStatePatch {
  auditError: null;
  auditLoading: false;
  jobRecords: StudioJobRecord[];
  jobStatus: StudioJobStatus;
}

export interface IdentityAccountsLoadedStatePatch {
  auditError: null;
  auditLoading: false;
  identityBrowserUsers: StudioIdentityBrowserUser[];
  identityServiceAccounts: StudioIdentityServiceAccount[];
}

export interface IdentityAccountsMutatedStatePatch extends IdentityAccountsLoadedStatePatch {
  auditExport: StudioAuditExport;
}

export interface OperatorStatusLoadedStatePatch extends JobStatusLoadedStatePatch {
  auditStatus: StudioOperatorStatus["audit"];
  operatorStatus: StudioOperatorStatus;
}

export function adminBusyState(): AdminBusyStatePatch {
  return {
    auditError: null,
    auditLoading: true,
  };
}

export function adminFailureState(
  error: unknown,
  fallbackMessage: string,
): AdminFailureStatePatch {
  return {
    auditError: error instanceof Error && error.message.length > 0
      ? error.message
      : fallbackMessage,
    auditLoading: false,
  };
}

export function jobStatusLoadedState(
  jobStatus: StudioJobStatus,
  jobList: StudioJobListResponse,
): JobStatusLoadedStatePatch {
  return {
    auditError: null,
    auditLoading: false,
    jobRecords: jobList.jobs,
    jobStatus,
  };
}

export function identityAccountsLoadedState(
  accountsResponse: StudioIdentityServiceAccountsResponse,
  usersResponse: StudioIdentityBrowserUsersResponse,
): IdentityAccountsLoadedStatePatch {
  return {
    auditError: null,
    auditLoading: false,
    identityBrowserUsers: usersResponse.browser_users,
    identityServiceAccounts: accountsResponse.service_accounts,
  };
}

export function identityAccountsMutatedState(
  accountsResponse: StudioIdentityServiceAccountsResponse,
  usersResponse: StudioIdentityBrowserUsersResponse,
  auditExport: StudioAuditExport,
): IdentityAccountsMutatedStatePatch {
  return {
    ...identityAccountsLoadedState(accountsResponse, usersResponse),
    auditExport,
  };
}

export function operatorStatusLoadedState(
  operatorStatus: StudioOperatorStatus,
  jobList: StudioJobListResponse,
): OperatorStatusLoadedStatePatch {
  return {
    auditError: null,
    auditLoading: false,
    auditStatus: operatorStatus.audit,
    jobRecords: jobList.jobs,
    jobStatus: operatorStatus.jobs,
    operatorStatus,
  };
}
