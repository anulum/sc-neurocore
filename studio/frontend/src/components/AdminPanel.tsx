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
    loadAuditExport,
    loadAuditStatus,
  } = useStudioStore();
  const model = buildAdminShellModel({
    auditError,
    auditExport,
    auditStatus,
    capabilities,
  });

  return (
    <AdminPanelView
      auditLoading={auditLoading}
      model={model}
      onLoadAuditExport={loadAuditExport}
      onLoadAuditStatus={loadAuditStatus}
    />
  );
}
