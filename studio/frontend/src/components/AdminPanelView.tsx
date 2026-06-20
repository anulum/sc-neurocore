import type { FormEvent } from "react";

import type { AdminShellModel } from "../adminShell";

export interface AdminPanelViewProps {
  auditLoading: boolean;
  model: AdminShellModel;
  onLoadAuditExport: () => Promise<void>;
  onLoadAuditStatus: () => Promise<void>;
  onLoadIdentityServiceAccounts: () => Promise<void>;
  onLoadJobStatus: () => Promise<void>;
  onLoadOperatorStatus: () => Promise<void>;
  onUpdateIdentityServiceAccount: (
    principalId: string,
    update: { active: boolean; expires_at_utc: string | null; roles: string[] },
  ) => Promise<void>;
  onUpdateIdentityBrowserUser: (
    username: string,
    update: { active: boolean; expires_at_utc: string | null; roles: string[] },
  ) => Promise<void>;
}

export default function AdminPanelView({
  auditLoading,
  model,
  onLoadAuditExport,
  onLoadAuditStatus,
  onLoadIdentityServiceAccounts,
  onLoadJobStatus,
  onLoadOperatorStatus,
  onUpdateIdentityBrowserUser,
  onUpdateIdentityServiceAccount,
}: AdminPanelViewProps) {
  function identityUpdateFromForm(form: FormData) {
    const rolesText = String(form.get("roles") ?? "");
    const roles = rolesText.split(",").map((role) => role.trim()).filter(Boolean);
    return {
      active: form.get("active") === "on",
      expires_at_utc: null,
      roles,
    };
  }

  function submitIdentityUpdate(event: FormEvent<HTMLFormElement>, principalId: string) {
    event.preventDefault();
    void onUpdateIdentityServiceAccount(
      principalId,
      identityUpdateFromForm(new FormData(event.currentTarget)),
    );
  }

  function submitBrowserUserUpdate(event: FormEvent<HTMLFormElement>, username: string) {
    event.preventDefault();
    void onUpdateIdentityBrowserUser(
      username,
      identityUpdateFromForm(new FormData(event.currentTarget)),
    );
  }

  return (
    <div className="admin-panel">
      <section className="admin-section">
        <div className="admin-section-header">
          <h2>Operator</h2>
          <div className="admin-actions">
            <button
              aria-label="Refresh operator status"
              onClick={() => void onLoadOperatorStatus()}
              disabled={auditLoading}
            >
              Status
            </button>
          </div>
        </div>
        <div className="admin-metrics">
          <div><span>Profile</span><strong>{model.operator.deploymentProfile}</strong></div>
          <div><span>Routes</span><strong>{model.operator.routePolicyLabel}</strong></div>
          <div><span>Identity</span><strong>{model.operator.identityMode}</strong></div>
          <div><span>EDA CPU</span><strong>{model.operator.edaCpuLimit}</strong></div>
          <div><span>EDA memory</span><strong>{model.operator.edaMemoryLimit}</strong></div>
          <div><span>EDA limits</span><strong>{model.operator.edaLimitSupport}</strong></div>
          <div><span>Job timeout</span><strong>{model.operator.jobTimeout}</strong></div>
          <div><span>Artifact cap</span><strong>{model.operator.jobArtifactLimit}</strong></div>
        </div>
        <div className="admin-audit-list">
          <div className="admin-audit-row">
            <span>schema</span>
            <strong>{model.operator.schemaVersion}</strong>
            <small>Operator status contract version</small>
          </div>
        </div>
      </section>

      <section className="admin-section">
        <div className="admin-section-header">
          <h2>Identity</h2>
          <div className="admin-actions">
            <button
              aria-label="Refresh identity accounts"
              onClick={() => void onLoadIdentityServiceAccounts()}
              disabled={auditLoading}
            >
              Accounts
            </button>
          </div>
        </div>
        <div className="admin-audit-list">
          {model.identityAccounts.length === 0 ? (
            <div className="admin-audit-row">
              <span>unavailable</span>
              <strong>No persistent accounts loaded</strong>
              <small>service-account store not returned by backend</small>
            </div>
          ) : model.identityAccounts.map((account) => (
            <form
              key={account.principalId}
              className="admin-audit-row admin-identity-row"
              onSubmit={(event) => submitIdentityUpdate(event, account.principalId)}
            >
              <span>{account.activeLabel}</span>
              <strong>{account.principalId}</strong>
              <label>
                Roles
                <input
                  aria-label={`${account.principalId} roles`}
                  name="roles"
                  defaultValue={account.rolesText}
                  disabled={auditLoading}
                />
              </label>
              <label>
                <input
                  aria-label={`${account.principalId} active`}
                  name="active"
                  type="checkbox"
                  defaultChecked={account.active}
                  disabled={auditLoading}
                />
                Active
              </label>
              <small>{account.expiresAt}</small>
              <button
                aria-label={`Save ${account.principalId} identity`}
                disabled={auditLoading}
                type="submit"
              >
                Save
              </button>
            </form>
          ))}
          {model.identityBrowserUsers.length === 0 ? (
            <div className="admin-audit-row">
              <span>unavailable</span>
              <strong>No browser users loaded</strong>
              <small>browser-user store not returned by backend</small>
            </div>
          ) : model.identityBrowserUsers.map((user) => (
            <form
              key={user.username}
              className="admin-audit-row admin-identity-row"
              onSubmit={(event) => submitBrowserUserUpdate(event, user.username)}
            >
              <span>{user.activeLabel}</span>
              <strong>{user.username}</strong>
              <label>
                Roles
                <input
                  aria-label={`${user.username} browser roles`}
                  name="roles"
                  defaultValue={user.rolesText}
                  disabled={auditLoading}
                />
              </label>
              <label>
                <input
                  aria-label={`${user.username} browser active`}
                  name="active"
                  type="checkbox"
                  defaultChecked={user.active}
                  disabled={auditLoading}
                />
                Active
              </label>
              <small>{user.principalId} - {user.expiresAt}</small>
              <button
                aria-label={`Save ${user.username} browser user`}
                disabled={auditLoading}
                type="submit"
              >
                Save
              </button>
            </form>
          ))}
        </div>
      </section>

      <section className="admin-section">
        <div className="admin-section-header">
          <h2>Audit</h2>
          <div className="admin-actions">
            <button
              aria-label="Refresh audit status"
              onClick={() => void onLoadAuditStatus()}
              disabled={auditLoading}
            >
              Status
            </button>
            <button
              aria-label="Export audit events"
              onClick={() => void onLoadAuditExport()}
              disabled={auditLoading}
            >
              Export
            </button>
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
          <h2>Jobs</h2>
          <div className="admin-actions">
            <button
              aria-label="Refresh job status"
              onClick={() => void onLoadJobStatus()}
              disabled={auditLoading}
            >
              Status
            </button>
          </div>
        </div>
        <div className="admin-metrics">
          <div><span>Health</span><strong>{model.jobs.healthLabel}</strong></div>
          <div><span>Active</span><strong>{model.jobs.active}</strong></div>
          <div><span>Completed</span><strong>{model.jobs.completed}</strong></div>
          <div><span>Timed out</span><strong>{model.jobs.timedOut}</strong></div>
        </div>
        <div className="admin-audit-list">
          <div className="admin-audit-row">
            <span>{model.jobs.configured ? "configured" : "unconfigured"}</span>
            <strong>{model.jobs.allowedKinds}</strong>
            <small>{model.jobs.failed} failed jobs recorded by the local worker manager</small>
          </div>
          {model.jobRecords.map((job) => (
            <div key={job.jobId} className="admin-audit-row">
              <span>{job.status}</span>
              <strong>{job.kind} - {job.jobId}</strong>
              <small>
                {job.owner} - {job.finishedAt} - {job.artifactCount} artifacts
                {job.error ? ` - ${job.error}` : ""}
              </small>
            </div>
          ))}
        </div>
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
