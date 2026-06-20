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
    identityBrowserUsers,
    identityServiceAccounts,
    jobRecords,
    jobStatus,
    operatorStatus,
    loadAuditExport,
    loadAuditStatus,
    loadIdentityServiceAccounts,
    loadJobStatus,
    loadOperatorStatus,
    updateIdentityBrowserUser,
    updateIdentityServiceAccount,
  } = useStudioStore();
  const model = buildAdminShellModel({
    auditError,
    auditExport,
    auditStatus,
    capabilities,
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
      onLoadAuditExport={loadAuditExport}
      onLoadAuditStatus={loadAuditStatus}
      onLoadIdentityServiceAccounts={loadIdentityServiceAccounts}
      onLoadJobStatus={loadJobStatus}
      onLoadOperatorStatus={loadOperatorStatus}
      onUpdateIdentityBrowserUser={updateIdentityBrowserUser}
      onUpdateIdentityServiceAccount={updateIdentityServiceAccount}
    />
  );
}
