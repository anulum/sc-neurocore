// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio admin store state helpers

/**
 * The store patches the Admin panel's own requests produce.
 *
 * They are separate from `auditShell.ts` because the panel loads more than the
 * audit trail: job status, identity accounts, and the operator status that
 * answers several of those at once. They share `auditError` and `auditLoading`
 * because the panel shows one busy state and one error line for all of it.
 *
 * Mutating an identity account replaces the audit export in the same patch.
 * That is the point of the mutation view: an operator who has just disabled an
 * account should see the audit line recording it, not the log as it stood
 * before the change.
 */
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

/** A panel request has started. */
export interface AdminBusyStatePatch {
  auditError: null;
  auditLoading: true;
}

/** A panel request failed, with the message to show. */
export interface AdminFailureStatePatch {
  auditError: string;
  auditLoading: false;
}

/** The job status and its records arrived together. */
export interface JobStatusLoadedStatePatch {
  auditError: null;
  auditLoading: false;
  jobRecords: StudioJobRecord[];
  jobStatus: StudioJobStatus;
}

/** Both identity listings arrived; they are always read as a pair. */
export interface IdentityAccountsLoadedStatePatch {
  auditError: null;
  auditLoading: false;
  identityBrowserUsers: StudioIdentityBrowserUser[];
  identityServiceAccounts: StudioIdentityServiceAccount[];
}

/**
 * An identity account changed. The audit export is replaced so the operator
 * sees the entry their own action wrote.
 */
export interface IdentityAccountsMutatedStatePatch extends IdentityAccountsLoadedStatePatch {
  auditExport: StudioAuditExport;
}

/**
 * The operator status arrived, which answers the audit status and the job
 * status too. Those are set from it rather than left to their own requests.
 */
export interface OperatorStatusLoadedStatePatch extends JobStatusLoadedStatePatch {
  auditStatus: StudioOperatorStatus["audit"];
  operatorStatus: StudioOperatorStatus;
}

/**
 * Start a panel request.
 *
 * @returns The patch.
 */
export function adminBusyState(): AdminBusyStatePatch {
  return {
    auditError: null,
    auditLoading: true,
  };
}

/**
 * Record a panel request that failed.
 *
 * @param error - What was thrown, which need not be an `Error`.
 * @param fallbackMessage - What to show when it carries no message.
 * @returns The patch.
 */
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

/**
 * Record the job status and its records.
 *
 * @param jobStatus - The queue's status.
 * @param jobList - The job listing from the same refresh.
 * @returns The patch.
 */
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

/**
 * Record both identity listings.
 *
 * @param accountsResponse - The service accounts.
 * @param usersResponse - The browser users.
 * @returns The patch.
 */
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

/**
 * Record an identity change, with the audit entry it wrote.
 *
 * @param accountsResponse - The service accounts after the change.
 * @param usersResponse - The browser users after the change.
 * @param auditExport - The audit export after the change.
 * @returns The patch.
 */
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

/**
 * Record the operator status and everything it answers.
 *
 * @param operatorStatus - The status.
 * @param jobList - The job listing from the same refresh.
 * @returns The patch.
 */
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
