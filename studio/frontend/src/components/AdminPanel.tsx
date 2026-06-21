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
