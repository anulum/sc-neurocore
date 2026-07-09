// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { buildAdminShellModel } from "../adminShell";
import { useStudioStore } from "../stores/studio";
import AdminPanelView from "./AdminPanelView";

export default function AdminPanel() {
  const {
    auditArchive,
    auditArchivePurge,
    auditArchiveRetention,
    auditArchiveRestore,
    auditArchiveValidation,
    auditError,
    auditExport,
    auditLoading,
    auditStatus,
    capabilities,
    createEvidenceBundle,
    downloadEvidenceBundleArtifact,
    evidenceBundle,
    evidenceBundleError,
    evidenceBundleLoading,
    identityBrowserUsers,
    identityServiceAccounts,
    jobRecords,
    jobStatus,
    operatorStatus,
    createIdentityBrowserUser,
    loadAuditExport,
    loadAuditStatus,
    loadIdentityServiceAccounts,
    loadJobStatus,
    loadOperatorStatus,
    rotateIdentityBrowserUserPassword,
    updateIdentityBrowserUser,
    updateIdentityServiceAccount,
    createAuditQuarantineArchive,
    loadAuditQuarantineArchiveRetention,
    purgeAuditQuarantineArchiveRetention,
    restoreAuditQuarantineArchive,
    validateAuditQuarantineArchive,
  } = useStudioStore();
  const model = buildAdminShellModel({
    auditArchive,
    auditArchivePurge,
    auditArchiveRetention,
    auditArchiveRestore,
    auditArchiveValidation,
    auditError,
    auditExport,
    auditStatus,
    capabilities,
    evidenceBundle,
    evidenceBundleError,
    evidenceBundleLoading,
    identityBrowserUsers,
    identityServiceAccounts,
    jobRecords,
    jobStatus,
    operatorStatus,
  });

  return (
    <AdminPanelView
      auditLoading={auditLoading}
      model={model}
      onCreateAuditArchive={createAuditQuarantineArchive}
      onCreateEvidenceBundle={createEvidenceBundle}
      onCreateIdentityBrowserUser={createIdentityBrowserUser}
      onDownloadEvidenceArtifact={downloadEvidenceBundleArtifact}
      onLoadAuditExport={loadAuditExport}
      onLoadAuditArchiveRetention={loadAuditQuarantineArchiveRetention}
      onLoadAuditStatus={loadAuditStatus}
      onLoadIdentityServiceAccounts={loadIdentityServiceAccounts}
      onLoadJobStatus={loadJobStatus}
      onLoadOperatorStatus={loadOperatorStatus}
      onPurgeAuditArchiveRetention={purgeAuditQuarantineArchiveRetention}
      onRestoreAuditArchive={restoreAuditQuarantineArchive}
      onRotateIdentityBrowserUserPassword={rotateIdentityBrowserUserPassword}
      onUpdateIdentityBrowserUser={updateIdentityBrowserUser}
      onUpdateIdentityServiceAccount={updateIdentityServiceAccount}
      onValidateAuditArchive={validateAuditQuarantineArchive}
    />
  );
}
