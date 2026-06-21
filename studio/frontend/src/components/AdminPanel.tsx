import { buildAdminShellModel } from "../adminShell";
import { useStudioStore } from "../stores/studio";
import AdminPanelView from "./AdminPanelView";

export default function AdminPanel() {
  const {
    auditArchive,
    auditArchivePurge,
    auditArchiveRetention,
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
  } = useStudioStore();
  const model = buildAdminShellModel({
    auditArchive,
    auditArchivePurge,
    auditArchiveRetention,
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
      onRotateIdentityBrowserUserPassword={rotateIdentityBrowserUserPassword}
      onUpdateIdentityBrowserUser={updateIdentityBrowserUser}
      onUpdateIdentityServiceAccount={updateIdentityServiceAccount}
    />
  );
}
