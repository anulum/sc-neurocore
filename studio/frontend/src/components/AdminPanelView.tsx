import { useState, type FormEvent } from "react";

import type { AdminShellModel } from "../adminShell";
import type { StudioEvidenceBundleRequest } from "../api/client";

export interface AdminPanelViewProps {
  auditLoading: boolean;
  model: AdminShellModel;
  onCreateIdentityBrowserUser: (
    create: {
      active: boolean;
      expires_at_utc: string | null;
      password: string;
      principal_id: string;
      roles: string[];
      username: string;
    },
  ) => Promise<void>;
  onCreateEvidenceBundle: (request: StudioEvidenceBundleRequest) => Promise<void>;
  onDownloadEvidenceArtifact: (relativePath: string) => Promise<void>;
  onLoadAuditExport: () => Promise<void>;
  onLoadAuditStatus: () => Promise<void>;
  onLoadIdentityServiceAccounts: () => Promise<void>;
  onLoadJobStatus: () => Promise<void>;
  onLoadOperatorStatus: () => Promise<void>;
  onRotateIdentityBrowserUserPassword: (username: string, password: string) => Promise<void>;
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
  onCreateEvidenceBundle,
  onCreateIdentityBrowserUser,
  onDownloadEvidenceArtifact,
  onLoadAuditExport,
  onLoadAuditStatus,
  onLoadIdentityServiceAccounts,
  onLoadJobStatus,
  onLoadOperatorStatus,
  onRotateIdentityBrowserUserPassword,
  onUpdateIdentityBrowserUser,
  onUpdateIdentityServiceAccount,
}: AdminPanelViewProps) {
  const [evidenceJobIds, setEvidenceJobIds] = useState("");

  function textList(value: FormDataEntryValue | null): string[] {
    return String(value ?? "")
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
  }

  function optionalText(value: FormDataEntryValue | null): string | null {
    const text = String(value ?? "").trim();
    return text.length > 0 ? text : null;
  }

  function boundedInteger(
    value: FormDataEntryValue | null,
    fallback: number,
    minimum: number,
    maximum: number,
  ): number {
    const parsed = Number(value ?? fallback);
    if (!Number.isFinite(parsed)) {
      return fallback;
    }
    return Math.min(Math.max(Math.trunc(parsed), minimum), maximum);
  }

  function jsonObjects(value: FormDataEntryValue | null): Record<string, unknown>[] {
    const text = String(value ?? "").trim();
    if (text.length === 0) {
      return [];
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      return [];
    }
    if (Array.isArray(parsed)) {
      return parsed.filter((item): item is Record<string, unknown> =>
        typeof item === "object" && item !== null && !Array.isArray(item),
      );
    }
    if (typeof parsed === "object" && parsed !== null) {
      return [parsed as Record<string, unknown>];
    }
    return [];
  }

  function identityUpdateFromForm(form: FormData) {
    return {
      active: form.get("active") === "on",
      expires_at_utc: null,
      roles: textList(form.get("roles")),
    };
  }

  function submitIdentityUpdate(event: FormEvent<HTMLFormElement>, principalId: string) {
    event.preventDefault();
    void onUpdateIdentityServiceAccount(
      principalId,
      identityUpdateFromForm(new FormData(event.currentTarget)),
    );
  }

  function submitBrowserUserCreate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const formElement = event.currentTarget;
    const form = new FormData(formElement);
    const rolesText = String(form.get("roles") ?? "");
    void (async () => {
      await onCreateIdentityBrowserUser({
        active: form.get("active") === "on",
        expires_at_utc: null,
        password: String(form.get("password") ?? ""),
        principal_id: String(form.get("principalId") ?? "").trim(),
        roles: rolesText.split(",").map((role) => role.trim()).filter(Boolean),
        username: String(form.get("username") ?? "").trim(),
      });
      formElement.reset();
      const activeInput = formElement.elements.namedItem("active");
      if (activeInput instanceof HTMLInputElement) {
        activeInput.checked = true;
      }
    })();
  }

  function submitBrowserUserUpdate(event: FormEvent<HTMLFormElement>, username: string) {
    event.preventDefault();
    const formElement = event.currentTarget;
    const form = new FormData(formElement);
    const nextSecret = String(form.get("newSecret") ?? "");
    void (async () => {
      await onUpdateIdentityBrowserUser(username, identityUpdateFromForm(form));
      if (nextSecret.length > 0) {
        await onRotateIdentityBrowserUserPassword(username, nextSecret);
        const secretInput = formElement.elements.namedItem("newSecret");
        if (secretInput instanceof HTMLInputElement) {
          secretInput.value = "";
        }
      }
    })();
  }

  function submitEvidenceBundle(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const method = optionalText(form.get("replayMethod"));
    const route = optionalText(form.get("replayRoute"));
    const requestSha256 = optionalText(form.get("requestSha256"));
    const note = optionalText(form.get("operatorNote"));
    const commandReplay: Record<string, unknown> = {};
    if (method !== null) {
      commandReplay.method = method;
    }
    if (route !== null) {
      commandReplay.route = route;
    }
    if (requestSha256 !== null) {
      commandReplay.request_sha256 = requestSha256;
    }
    if (note !== null) {
      commandReplay.note = note;
    }

    void onCreateEvidenceBundle({
      audit_limit: boundedInteger(form.get("auditLimit"), 100, 1, 1000),
      analysis_results: jsonObjects(form.get("analysisResults")),
      command_replay: Object.keys(commandReplay).length > 0 ? commandReplay : null,
      default_flow_attestations: jsonObjects(form.get("defaultFlowAttestations")),
      default_flow_runs: jsonObjects(form.get("defaultFlowRuns")),
      include_audit: form.get("includeAudit") === "on",
      job_ids: textList(form.get("jobIds")),
      project_name: optionalText(form.get("projectName")),
      simulation_results: jsonObjects(form.get("simulationResults")),
    });
  }

  function addEvidenceJobId(jobId: string) {
    setEvidenceJobIds((current) => {
      const selected = current
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean);
      if (!selected.includes(jobId)) {
        selected.push(jobId);
      }
      return selected.join(", ");
    });
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
          <div><span>Route inventory</span><strong>{model.operator.routePolicyInventory}</strong></div>
          <div><span>Route audit</span><strong>{model.operator.routePolicyAuditLabel}</strong></div>
          <div><span>Identity</span><strong>{model.operator.identityMode}</strong></div>
          <div><span>EDA CPU</span><strong>{model.operator.edaCpuLimit}</strong></div>
          <div><span>EDA memory</span><strong>{model.operator.edaMemoryLimit}</strong></div>
          <div><span>EDA limits</span><strong>{model.operator.edaLimitSupport}</strong></div>
          <div><span>Job timeout</span><strong>{model.operator.jobTimeout}</strong></div>
          <div><span>Artifact cap</span><strong>{model.operator.jobArtifactLimit}</strong></div>
          <div><span>Login limit</span><span style={{ fontWeight: 700 }}>{model.operator.browserLoginLimit}</span></div>
          <div><span>Login window</span><span style={{ fontWeight: 700 }}>{model.operator.browserLoginWindow}</span></div>
          <div><span>Login cooldown</span><span style={{ fontWeight: 700 }}>{model.operator.browserLoginCooldown}</span></div>
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
          <form
            className="admin-audit-row admin-identity-row"
            onSubmit={submitBrowserUserCreate}
          >
            <span>new</span>
            <strong>Browser user</strong>
            <label>
              Username
              <input
                aria-label="New browser username"
                name="username"
                disabled={auditLoading}
                required
              />
            </label>
            <label>
              Principal
              <input
                aria-label="New browser principal"
                name="principalId"
                disabled={auditLoading}
                required
              />
            </label>
            <label>
              Roles
              <input
                aria-label="New browser roles"
                name="roles"
                defaultValue="studio.viewer"
                disabled={auditLoading}
                required
              />
            </label>
            <label>
              Secret
              <input
                aria-label="New browser secret"
                autoComplete="new-password"
                name="password"
                type="password"
                disabled={auditLoading}
                required
              />
            </label>
            <label>
              <input
                aria-label="New browser active"
                name="active"
                type="checkbox"
                defaultChecked
                disabled={auditLoading}
              />
              Active
            </label>
            <small>pending</small>
            <button
              aria-label="Create browser user"
              disabled={auditLoading}
              type="submit"
            >
              Create
            </button>
          </form>
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
              <label>
                New secret
                <input
                  aria-label={`${user.username} new secret`}
                  autoComplete="new-password"
                  name="newSecret"
                  type="password"
                  disabled={auditLoading}
                />
              </label>
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
          <div><span>Identity lifecycle</span><span style={{ fontWeight: 700 }}>{model.audit.identityLifecycle}</span></div>
          <div><span>Identity allowed</span><span style={{ fontWeight: 700 }}>{model.audit.identityLifecycleAllowed}</span></div>
          <div><span>Identity denied</span><span style={{ fontWeight: 700 }}>{model.audit.identityLifecycleDenied}</span></div>
          <div><span>Latest identity</span><span style={{ fontWeight: 700 }}>{model.audit.latestIdentityLifecycleAction ?? "none"}</span></div>
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
          {model.jobs.resourceProfiles.map((profile) => (
            <div key={profile} className="admin-audit-row">
              <span>profile</span>
              <strong>{profile}</strong>
              <small>default timeout, artifact ceiling, and execution model</small>
            </div>
          ))}
          {model.jobRecords.map((job) => (
            <div key={job.jobId} className="admin-audit-row admin-job-row">
              <span>{job.status}</span>
              <strong>{job.kind} - {job.jobId}</strong>
              <small>
                {job.owner} - {job.finishedAt} - {job.artifactCount} artifacts
                {" - "}{job.evidenceArtifactCount} evidence
                {job.error ? ` - ${job.error}` : ""}
              </small>
              <small title={job.artifactPaths}>{job.artifactPaths}</small>
              <button
                aria-label={`Add ${job.jobId} to evidence bundle`}
                disabled={model.evidenceBundle.loading}
                onClick={() => addEvidenceJobId(job.jobId)}
                type="button"
              >
                Bundle
              </button>
            </div>
          ))}
        </div>
      </section>

      <section className="admin-section admin-evidence-section">
        <div className="admin-section-header">
          <h2>Evidence</h2>
          <span>{model.evidenceBundle.loading ? "exporting" : "ready"}</span>
        </div>
        <div className="admin-metrics">
          <div><span>Bundle</span><strong>{model.evidenceBundle.bundleId}</strong></div>
          <div><span>Job</span><strong>{model.evidenceBundle.jobId}</strong></div>
          <div><span>Artifacts</span><strong>{model.evidenceBundle.artifactCount}</strong></div>
          <div><span>Entries</span><strong>{model.evidenceBundle.manifestEntryCount}</strong></div>
          <div><span>Types</span><strong>{model.evidenceBundle.entryTypes}</strong></div>
          <div><span>Classes</span><strong>{model.evidenceBundle.evidenceClasses}</strong></div>
          <div><span>Sources</span><strong>{model.evidenceBundle.sourceJobs}</strong></div>
        </div>
        {model.evidenceBundle.error && (
          <div className="admin-warning">{model.evidenceBundle.error}</div>
        )}
        {model.evidenceBundle.entries.length > 0 && (
          <div className="admin-audit-list">
            {model.evidenceBundle.entries.map((entry) => (
              <div
                key={`${entry.index}-${entry.type}-${entry.detail}`}
                className="admin-audit-row admin-manifest-row"
              >
                <span>{entry.type}</span>
                <strong title={entry.source}>{entry.source}</strong>
                <small>{entry.classification} - {entry.detail}</small>
              </div>
            ))}
          </div>
        )}
        {model.evidenceBundle.artifacts.length > 0 && (
          <div className="admin-audit-list">
            {model.evidenceBundle.artifacts.map((artifact) => (
              <div key={artifact.relativePath} className="admin-audit-row admin-artifact-row">
                <span>file</span>
                <strong title={artifact.relativePath}>{artifact.relativePath}</strong>
                <small>{artifact.sizeLabel} - sha {artifact.sha256Label}</small>
                <button
                  aria-label={`Download evidence artifact ${artifact.relativePath}`}
                  disabled={model.evidenceBundle.loading}
                  onClick={() => void onDownloadEvidenceArtifact(artifact.relativePath)}
                  type="button"
                >
                  Download
                </button>
              </div>
            ))}
          </div>
        )}
        <form className="admin-evidence-form" onSubmit={submitEvidenceBundle}>
          <label>
            Project
            <input
              aria-label="Evidence project name"
              name="projectName"
              disabled={model.evidenceBundle.loading}
            />
          </label>
          <label>
            Job IDs
            <input
              aria-label="Evidence job IDs"
              name="jobIds"
              onChange={(event) => setEvidenceJobIds(event.currentTarget.value)}
              value={evidenceJobIds}
              disabled={model.evidenceBundle.loading}
            />
          </label>
          <label className="admin-evidence-wide">
            Simulation JSON
            <textarea
              aria-label="Evidence simulation JSON"
              name="simulationResults"
              disabled={model.evidenceBundle.loading}
              rows={4}
            />
          </label>
          <label className="admin-evidence-wide">
            Analysis JSON
            <textarea
              aria-label="Evidence analysis JSON"
              name="analysisResults"
              disabled={model.evidenceBundle.loading}
              rows={4}
            />
          </label>
          <label className="admin-evidence-wide">
            Flow Run JSON
            <textarea
              aria-label="Evidence default-flow run JSON"
              name="defaultFlowRuns"
              disabled={model.evidenceBundle.loading}
              rows={4}
            />
          </label>
          <label className="admin-evidence-wide">
            Flow Attestation JSON
            <textarea
              aria-label="Evidence default-flow attestation JSON"
              name="defaultFlowAttestations"
              disabled={model.evidenceBundle.loading}
              rows={4}
            />
          </label>
          <label>
            Audit
            <input
              aria-label="Include audit export"
              name="includeAudit"
              type="checkbox"
              defaultChecked
              disabled={model.evidenceBundle.loading}
            />
          </label>
          <label>
            Limit
            <input
              aria-label="Evidence audit limit"
              name="auditLimit"
              type="number"
              min={1}
              max={1000}
              defaultValue={100}
              disabled={model.evidenceBundle.loading}
            />
          </label>
          <label>
            Method
            <input
              aria-label="Evidence replay method"
              name="replayMethod"
              defaultValue="POST"
              disabled={model.evidenceBundle.loading}
            />
          </label>
          <label>
            Route
            <input
              aria-label="Evidence replay route"
              name="replayRoute"
              disabled={model.evidenceBundle.loading}
            />
          </label>
          <label>
            Request SHA
            <input
              aria-label="Evidence request SHA-256"
              name="requestSha256"
              disabled={model.evidenceBundle.loading}
            />
          </label>
          <label>
            Note
            <input
              aria-label="Evidence operator note"
              name="operatorNote"
              disabled={model.evidenceBundle.loading}
            />
          </label>
          <button
            aria-label="Create evidence bundle"
            className="admin-evidence-submit"
            disabled={model.evidenceBundle.loading}
            type="submit"
          >
            Export
          </button>
        </form>
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
