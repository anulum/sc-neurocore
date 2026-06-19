import type { AdminShellModel } from "../adminShell";

export interface AdminPanelViewProps {
  auditLoading: boolean;
  model: AdminShellModel;
  onLoadAuditExport: () => Promise<void>;
  onLoadAuditStatus: () => Promise<void>;
}

export default function AdminPanelView({
  auditLoading,
  model,
  onLoadAuditExport,
  onLoadAuditStatus,
}: AdminPanelViewProps) {
  return (
    <div className="admin-panel">
      <section className="admin-section">
        <div className="admin-section-header">
          <h2>Audit</h2>
          <div className="admin-actions">
            <button onClick={() => void onLoadAuditStatus()} disabled={auditLoading}>Status</button>
            <button onClick={() => void onLoadAuditExport()} disabled={auditLoading}>Export</button>
          </div>
        </div>
        <div className="admin-metrics">
          <div><span>Sink</span><strong>{model.audit.sinkType}</strong></div>
          <div><span>Health</span><strong>{model.audit.healthLabel}</strong></div>
          <div><span>Events</span><strong>{model.audit.total}</strong></div>
          <div><span>Denied</span><strong>{model.audit.denied}</strong></div>
        </div>
        {model.audit.error && <div className="admin-warning">{model.audit.error}</div>}
        {model.audit.lastError && <div className="admin-warning">{model.audit.lastError}</div>}
        {model.recentAuditEvents.length > 0 && (
          <div className="admin-audit-list">
            {model.recentAuditEvents.map((event) => (
              <div key={`${event.event_hash ?? event.action}-${event.timestamp_utc ?? "pending"}`}
                className="admin-audit-row">
                <span>{event.decision}</span>
                <strong>{event.action}</strong>
                <small>{event.principal_id ?? "anonymous"} - {event.reason}</small>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="admin-section">
        <div className="admin-section-header">
          <h2>Capabilities</h2>
          <span>{model.capabilities.registered} registered</span>
        </div>
        <div className="admin-audit-list">
          {model.unhealthyCapabilities.length === 0 ? (
            <div className="admin-audit-row">
              <span>ready</span>
              <strong>All registered capabilities healthy</strong>
              <small>Capability registry loaded from backend contract</small>
            </div>
          ) : model.unhealthyCapabilities.map((capability) => (
            <div key={capability.capability_id} className="admin-audit-row">
              <span>{capability.status}</span>
              <strong>{capability.title}</strong>
              <small>{capability.message}</small>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
