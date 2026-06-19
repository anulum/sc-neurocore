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
    jobRecords,
    jobStatus,
    operatorStatus,
    loadAuditExport,
    loadAuditStatus,
    loadJobStatus,
    loadOperatorStatus,
  } = useStudioStore();
  const model = buildAdminShellModel({
    auditError,
    auditExport,
    auditStatus,
    capabilities,
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
      onLoadJobStatus={loadJobStatus}
      onLoadOperatorStatus={loadOperatorStatus}
    />
  );
}
