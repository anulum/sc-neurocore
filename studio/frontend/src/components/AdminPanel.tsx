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
    identityServiceAccounts,
    jobRecords,
    jobStatus,
    operatorStatus,
    loadAuditExport,
    loadAuditStatus,
    loadIdentityServiceAccounts,
    loadJobStatus,
    loadOperatorStatus,
    updateIdentityServiceAccount,
  } = useStudioStore();
  const model = buildAdminShellModel({
    auditError,
    auditExport,
    auditStatus,
    capabilities,
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
      onUpdateIdentityServiceAccount={updateIdentityServiceAccount}
    />
  );
}
