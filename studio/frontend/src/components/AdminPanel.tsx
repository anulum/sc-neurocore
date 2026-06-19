import { summarizeAuditExport } from "../auditShell";
import { useStudioStore } from "../stores/studio";

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
  const auditSummary = summarizeAuditExport(auditExport);
  const unhealthyCapabilities = capabilities.filter((capability) => !capability.healthy);

  return (
    <div className="admin-panel">
      <section className="admin-section">
        <div className="admin-section-header">
          <h2>Audit</h2>
          <div className="admin-actions">
            <button onClick={() => void loadAuditStatus()} disabled={auditLoading}>Status</button>
            <button onClick={() => void loadAuditExport()} disabled={auditLoading}>Export</button>
          </div>
        </div>
        <div className="admin-metrics">
          <div><span>Sink</span><strong>{auditStatus?.sink_type ?? auditSummary.sinkType}</strong></div>
          <div><span>Health</span><strong>{auditStatus?.healthy === false ? "unhealthy" : "ready"}</strong></div>
          <div><span>Events</span><strong>{auditSummary.total}</strong></div>
          <div><span>Denied</span><strong>{auditSummary.denied}</strong></div>
        </div>
        {auditError && <div className="admin-warning">{auditError}</div>}
        {auditStatus?.last_error && <div className="admin-warning">{auditStatus.last_error}</div>}
        {auditExport && (
          <div className="admin-audit-list">
            {auditExport.events.slice(-8).reverse().map((event) => (
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
          <span>{capabilities.length} registered</span>
        </div>
        <div className="admin-audit-list">
          {unhealthyCapabilities.length === 0 ? (
            <div className="admin-audit-row">
              <span>ready</span>
              <strong>All registered capabilities healthy</strong>
              <small>Capability registry loaded from backend contract</small>
            </div>
          ) : unhealthyCapabilities.map((capability) => (
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
