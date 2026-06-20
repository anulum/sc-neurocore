import { buildAdminShellModel } from "../adminShell";
import { useStudioStore } from "../stores/studio";
import AdminPanelView from "./AdminPanelView";

export default function AdminPanel() {
  const {
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
  } = useStudioStore();
  const model = buildAdminShellModel({
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
      onCreateEvidenceBundle={createEvidenceBundle}
      onCreateIdentityBrowserUser={createIdentityBrowserUser}
      onDownloadEvidenceArtifact={downloadEvidenceBundleArtifact}
      onLoadAuditExport={loadAuditExport}
      onLoadAuditStatus={loadAuditStatus}
      onLoadIdentityServiceAccounts={loadIdentityServiceAccounts}
      onLoadJobStatus={loadJobStatus}
      onLoadOperatorStatus={loadOperatorStatus}
      onRotateIdentityBrowserUserPassword={rotateIdentityBrowserUserPassword}
      onUpdateIdentityBrowserUser={updateIdentityBrowserUser}
      onUpdateIdentityServiceAccount={updateIdentityServiceAccount}
    />
  );
}
