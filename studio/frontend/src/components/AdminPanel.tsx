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
    jobStatus,
    loadAuditExport,
    loadAuditStatus,
    loadJobStatus,
  } = useStudioStore();
  const model = buildAdminShellModel({
    auditError,
    auditExport,
    auditStatus,
    capabilities,
    jobStatus,
  });

  return (
    <AdminPanelView
      auditLoading={auditLoading}
      model={model}
      onLoadAuditExport={loadAuditExport}
      onLoadAuditStatus={loadAuditStatus}
      onLoadJobStatus={loadJobStatus}
    />
  );
}
