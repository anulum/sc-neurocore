// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio operator administration actions

import { createStoreArtifactDownloader } from "./studioArtifactDownload";
import {
  createStudioAuditQuarantineArchive,
  fetchStudioIdentityServiceAccounts,
  fetchStudioAuditQuarantineArchiveRetention,
  fetchStudioAuditExport,
  fetchStudioAuditStatus,
  fetchStudioJobs,
  fetchStudioJobStatus,
  fetchStudioOperatorStatus,
  fetchStudioIdentityBrowserUsers,
  purgeStudioAuditQuarantineArchiveRetention,
  restoreStudioAuditQuarantineArchive,
  rotateStudioIdentityBrowserUserPassword,
  createStudioIdentityBrowserUser,
  updateStudioIdentityBrowserUser,
  updateStudioIdentityServiceAccount,
  validateStudioAuditQuarantineArchive,
} from "../api/client";
import {
  auditArchiveCreatedState,
  auditArchivePurgedState,
  auditArchiveRestoredState,
  auditArchiveRetentionLoadedState,
  auditArchiveValidationLoadedState,
  auditExportLoadedState,
  auditFailureState,
  auditLoadingState,
  auditStatusLoadedState,
} from "../auditShell";
import {
  adminBusyState,
  adminFailureState,
  identityAccountsLoadedState,
  identityAccountsMutatedState,
  jobStatusLoadedState,
  operatorStatusLoadedState,
} from "../adminStoreState";
import { runStoreBundle } from "./studioBundle";
import type { StudioState, StudioStoreActions } from "./studioTypes";

/** The operator's administration of the Studio server. */
export type StudioAdminActions = Pick<StudioStoreActions,
  "loadAuditStatus" | "loadAuditExport" | "createAuditQuarantineArchive" | "loadAuditQuarantineArchiveRetention" | "validateAuditQuarantineArchive" | "restoreAuditQuarantineArchive" | "purgeAuditQuarantineArchiveRetention" | "createEvidenceBundle" | "createEvidenceBundleForSurface" | "downloadEvidenceBundleArtifact" | "downloadEvidenceBundleArtifactForSurface" | "loadJobStatus" | "loadIdentityServiceAccounts" | "createIdentityBrowserUser" | "updateIdentityServiceAccount" | "updateIdentityBrowserUser" | "rotateIdentityBrowserUserPassword" | "loadOperatorStatus">;

/**
 * Bind these actions to the Studio store.
 *
 * The operator's administration of the Studio server. The audit log and its quarantine archives are read, exported, validated, restored and purged; evidence bundles are built and downloaded; the job and operator status are read; and the identity service's accounts are listed, created, updated and given new passwords. Every action asks the server and writes the answer, or the refusal, into the store.
 *
 * @param set - Writes a patch into the store.
 * @param get - Reads the store as it stands.
 * @returns The actions.
 */
export function createStudioAdminActions(
  set: (partial: Partial<StudioState> | ((state: StudioState) => Partial<StudioState>)) => void,
  get: () => StudioState,
): StudioAdminActions {
  return {
  loadAuditStatus: async () => {
    set(auditLoadingState());
    try {
      const auditStatus = await fetchStudioAuditStatus();
      set(auditStatusLoadedState(auditStatus));
    } catch (error: unknown) {
      set(auditFailureState(error, "Audit status check failed"));
    }
  },
  loadAuditExport: async () => {
    set(auditLoadingState());
    try {
      const auditExport = await fetchStudioAuditExport(100);
      set(auditExportLoadedState(auditExport));
    } catch (error: unknown) {
      set(auditFailureState(error, "Audit export failed"));
    }
  },
  createAuditQuarantineArchive: async (limit) => {
    set(auditLoadingState());
    try {
      const auditArchive = await createStudioAuditQuarantineArchive(limit);
      const [operatorStatus, jobList, auditExport] = await Promise.all([
        fetchStudioOperatorStatus(),
        fetchStudioJobs(),
        fetchStudioAuditExport(100),
      ]);
      set(auditArchiveCreatedState(auditArchive, auditExport, operatorStatus, jobList));
    } catch (error: unknown) {
      set(auditFailureState(error, "Audit archive creation failed"));
    }
  },
  loadAuditQuarantineArchiveRetention: async (retainLatest) => {
    set(auditLoadingState());
    try {
      const auditArchiveRetention = await fetchStudioAuditQuarantineArchiveRetention(retainLatest);
      set(auditArchiveRetentionLoadedState(auditArchiveRetention));
    } catch (error: unknown) {
      set(auditFailureState(error, "Audit archive retention check failed"));
    }
  },
  validateAuditQuarantineArchive: async (archive, manifest) => {
    set(auditLoadingState());
    try {
      const auditArchiveValidation = await validateStudioAuditQuarantineArchive(archive, manifest);
      set(auditArchiveValidationLoadedState(auditArchiveValidation));
    } catch (error: unknown) {
      set(auditFailureState(error, "Audit archive validation failed"));
    }
  },
  restoreAuditQuarantineArchive: async (archive, manifest) => {
    set(auditLoadingState());
    try {
      const auditArchiveRestore = await restoreStudioAuditQuarantineArchive(archive, manifest);
      const [operatorStatus, jobList] = await Promise.all([
        fetchStudioOperatorStatus(),
        fetchStudioJobs(),
      ]);
      set(auditArchiveRestoredState(auditArchiveRestore, operatorStatus, jobList));
    } catch (error: unknown) {
      set(auditFailureState(error, "Audit archive restore failed"));
    }
  },
  purgeAuditQuarantineArchiveRetention: async (retainLatest) => {
    set(auditLoadingState());
    try {
      const auditArchivePurge = await purgeStudioAuditQuarantineArchiveRetention(retainLatest);
      const [operatorStatus, jobList, auditArchiveRetention] = await Promise.all([
        fetchStudioOperatorStatus(),
        fetchStudioJobs(),
        fetchStudioAuditQuarantineArchiveRetention(retainLatest),
      ]);
      set(auditArchivePurgedState(
        auditArchivePurge,
        auditArchiveRetention,
        operatorStatus,
        jobList,
      ));
    } catch (error: unknown) {
      set(auditFailureState(error, "Audit archive retention purge failed"));
    }
  },
  createEvidenceBundle: (request) => runStoreBundle("admin", request, get, set),
  createEvidenceBundleForSurface: (surface, request) => runStoreBundle(surface, request, get, set),
  downloadEvidenceBundleArtifact: async (relativePath) => {
    await get().downloadEvidenceBundleArtifactForSurface("admin", relativePath);
  },
  downloadEvidenceBundleArtifactForSurface: createStoreArtifactDownloader(get, set),
  loadJobStatus: async () => {
    set(adminBusyState());
    try {
      const [jobStatus, jobList] = await Promise.all([
        fetchStudioJobStatus(),
        fetchStudioJobs(),
      ]);
      set(jobStatusLoadedState(jobStatus, jobList));
    } catch (error: unknown) {
      set(adminFailureState(error, "Job status check failed"));
    }
  },
  loadIdentityServiceAccounts: async () => {
    set(adminBusyState());
    try {
      const [accountsResponse, usersResponse] = await Promise.all([
        fetchStudioIdentityServiceAccounts(),
        fetchStudioIdentityBrowserUsers(),
      ]);
      set(identityAccountsLoadedState(accountsResponse, usersResponse));
    } catch (error: unknown) {
      set(adminFailureState(error, "Identity account check failed"));
    }
  },
  createIdentityBrowserUser: async (create) => {
    set(adminBusyState());
    try {
      await createStudioIdentityBrowserUser(create);
      const [accountsResponse, usersResponse, auditExport] = await Promise.all([
        fetchStudioIdentityServiceAccounts(),
        fetchStudioIdentityBrowserUsers(),
        fetchStudioAuditExport(100),
      ]);
      set(identityAccountsMutatedState(accountsResponse, usersResponse, auditExport));
    } catch (error: unknown) {
      set(adminFailureState(error, "Browser user creation failed"));
    }
  },
  updateIdentityServiceAccount: async (principalId, update) => {
    set(adminBusyState());
    try {
      await updateStudioIdentityServiceAccount(principalId, update);
      const [accountsResponse, usersResponse, auditExport] = await Promise.all([
        fetchStudioIdentityServiceAccounts(),
        fetchStudioIdentityBrowserUsers(),
        fetchStudioAuditExport(100),
      ]);
      set(identityAccountsMutatedState(accountsResponse, usersResponse, auditExport));
    } catch (error: unknown) {
      set(adminFailureState(error, "Identity account update failed"));
    }
  },
  updateIdentityBrowserUser: async (username, update) => {
    set(adminBusyState());
    try {
      await updateStudioIdentityBrowserUser(username, update);
      const [accountsResponse, usersResponse, auditExport] = await Promise.all([
        fetchStudioIdentityServiceAccounts(),
        fetchStudioIdentityBrowserUsers(),
        fetchStudioAuditExport(100),
      ]);
      set(identityAccountsMutatedState(accountsResponse, usersResponse, auditExport));
    } catch (error: unknown) {
      set(adminFailureState(error, "Browser user update failed"));
    }
  },
  rotateIdentityBrowserUserPassword: async (username, password) => {
    set(adminBusyState());
    try {
      await rotateStudioIdentityBrowserUserPassword(username, { password });
      const [accountsResponse, usersResponse, auditExport] = await Promise.all([
        fetchStudioIdentityServiceAccounts(),
        fetchStudioIdentityBrowserUsers(),
        fetchStudioAuditExport(100),
      ]);
      set(identityAccountsMutatedState(accountsResponse, usersResponse, auditExport));
    } catch (error: unknown) {
      set(adminFailureState(error, "Browser user secret rotation failed"));
    }
  },
  loadOperatorStatus: async () => {
    set(adminBusyState());
    try {
      const [operatorStatus, jobList] = await Promise.all([
        fetchStudioOperatorStatus(),
        fetchStudioJobs(),
      ]);
      set(operatorStatusLoadedState(operatorStatus, jobList));
    } catch (error: unknown) {
      set(adminFailureState(error, "Operator status check failed"));
    }
  },
  };
}
