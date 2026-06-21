import { useState, type FormEvent } from "react";

import type { AdminAuditArchiveModel } from "../adminShell";

export interface AdminAuditArchiveSectionProps {
  auditLoading: boolean;
  archive: AdminAuditArchiveModel;
  onCreateAuditArchive: (limit: number) => Promise<void>;
  onLoadAuditArchiveRetention: (retainLatest: number) => Promise<void>;
  onPurgeAuditArchiveRetention: (retainLatest: number) => Promise<void>;
}

export default function AdminAuditArchiveSection({
  archive,
  auditLoading,
  onCreateAuditArchive,
  onLoadAuditArchiveRetention,
  onPurgeAuditArchiveRetention,
}: AdminAuditArchiveSectionProps) {
  const [retainLatest, setRetainLatest] = useState(String(archive.retainLatest));

  function boundedInteger(value: FormDataEntryValue | null, fallback: number): number {
    const parsed = Number(value ?? fallback);
    if (!Number.isFinite(parsed)) {
      return fallback;
    }
    return Math.min(Math.max(Math.trunc(parsed), 1), 1000);
  }

  function submitArchive(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    void onCreateAuditArchive(boundedInteger(form.get("archiveLimit"), 100));
  }

  function submitRetention(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const nextRetainLatest = boundedInteger(form.get("retainLatest"), archive.retainLatest);
    setRetainLatest(String(nextRetainLatest));
    void onLoadAuditArchiveRetention(nextRetainLatest);
  }

  function submitPurge(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const nextRetainLatest = boundedInteger(form.get("retainLatest"), archive.retainLatest);
    setRetainLatest(String(nextRetainLatest));
    void onPurgeAuditArchiveRetention(nextRetainLatest);
  }

  return (
    <section className="admin-section">
      <div className="admin-section-header">
        <h2>Audit archive</h2>
        <span>{archive.archiveCount} archives</span>
      </div>
      <div className="admin-metrics">
        <div><span>Latest archive</span><strong>{archive.archiveId}</strong></div>
        <div><span>Archived rows</span><strong>{archive.archivedEventCount}</strong></div>
        <div><span>Archive artifacts</span><strong>{archive.artifactCount}</strong></div>
        <div><span>Reasons</span><strong>{archive.reasonCounts}</strong></div>
        <div><span>Retain</span><strong>{archive.retainCount}</strong></div>
        <div><span>Prune candidates</span><strong>{archive.pruneCandidateCount}</strong></div>
        <div><span>Skipped</span><strong>{archive.skippedRecordCount}</strong></div>
        <div><span>Last purge</span><strong>{archive.lastPurge}</strong></div>
      </div>
      {archive.error && <div className="admin-warning">{archive.error}</div>}
      <div className="admin-audit-list">
        {archive.latestEntries.length === 0 ? (
          <div className="admin-audit-row">
            <span>unavailable</span>
            <strong>No archive retention inventory loaded</strong>
            <small>review retention before executing a purge</small>
          </div>
        ) : archive.latestEntries.map((entry) => (
          <div key={entry.jobId} className="admin-audit-row">
            <span>{entry.disposition}</span>
            <strong>{entry.archiveId}</strong>
            <small>
              {entry.jobId} - {entry.eventCount} rows - {entry.retainedEventCount} retained -
              {" "}{entry.finishedAt}
            </small>
          </div>
        ))}
      </div>
      <form className="admin-evidence-form" onSubmit={submitArchive}>
        <label>
          Archive limit
          <input
            aria-label="Audit archive limit"
            defaultValue={100}
            disabled={auditLoading}
            max={1000}
            min={1}
            name="archiveLimit"
            type="number"
          />
        </label>
        <button
          aria-label="Create audit quarantine archive"
          disabled={auditLoading}
          type="submit"
        >
          Archive
        </button>
      </form>
      <form className="admin-evidence-form" onSubmit={submitRetention}>
        <label>
          Retain latest
          <input
            aria-label="Audit archive retain latest"
            disabled={auditLoading}
            max={1000}
            min={1}
            name="retainLatest"
            onChange={(event) => setRetainLatest(event.currentTarget.value)}
            type="number"
            value={retainLatest}
          />
        </label>
        <button
          aria-label="Review audit archive retention"
          disabled={auditLoading}
          type="submit"
        >
          Review
        </button>
      </form>
      <form className="admin-evidence-form" onSubmit={submitPurge}>
        <input name="retainLatest" type="hidden" value={retainLatest} />
        <button
          aria-label="Purge audit archive prune candidates"
          disabled={auditLoading || archive.pruneCandidateCount === 0}
          type="submit"
        >
          Purge
        </button>
      </form>
    </section>
  );
}
