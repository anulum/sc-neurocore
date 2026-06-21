import type { StudioJobArtifact } from "../api/client";
import type { ProjectEvidenceModel } from "../projectEvidence";
import EvidenceSummaryStrip from "./EvidenceSummaryStrip";

export interface ProjectEvidenceStripProps {
  artifacts: StudioJobArtifact[];
  evidence: ProjectEvidenceModel;
  exportBundleId: string | null;
  exportError: string | null;
  exportJobId: string | null;
  loading: boolean;
  onDownloadArtifact: (relativePath: string) => void;
  onExportBundle: () => void;
}

function formatArtifactSize(sizeBytes: number): string {
  if (sizeBytes < 1024) return `${sizeBytes} B`;
  if (sizeBytes < 1024 * 1024) return `${(sizeBytes / 1024).toFixed(1)} KiB`;
  return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MiB`;
}

export default function ProjectEvidenceStrip({
  artifacts,
  evidence,
  exportBundleId,
  exportError,
  exportJobId,
  loading,
  onDownloadArtifact,
  onExportBundle,
}: ProjectEvidenceStripProps) {
  return (
    <div>
      <EvidenceSummaryStrip
        variant="panel"
        items={[
          { label: "class", value: evidence.classification },
          { label: "name", value: evidence.name },
          { label: "state sha", value: evidence.stateDigest },
          { label: "project sha", value: evidence.projectDigest },
          { label: "schema", value: evidence.schemaVersion },
        ]}
      />
      <div style={{ display: "flex", gap: 4, marginTop: 4, alignItems: "center", flexWrap: "wrap" }}>
        <button
          aria-label={`Export ${evidence.name} project evidence bundle`}
          disabled={loading}
          onClick={onExportBundle}
          style={{
            fontSize: 10,
            padding: "2px 6px",
            background: "var(--bg-tertiary)",
            color: "var(--text-secondary)",
            border: "1px solid var(--border)",
            borderRadius: 3,
            cursor: loading ? "wait" : "pointer",
          }}
          type="button"
        >
          Bundle
        </button>
        {exportBundleId !== null && (
          <span style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)", fontSize: 9 }}>
            {exportBundleId}
          </span>
        )}
        {exportJobId !== null && (
          <span style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)", fontSize: 9 }}>
            {exportJobId}
          </span>
        )}
        {exportError !== null && (
          <span style={{ color: "#ff5252", fontSize: 9 }}>{exportError}</span>
        )}
      </div>
      {artifacts.length > 0 && (
        <div
          aria-label="Project evidence bundle artifacts"
          style={{ display: "grid", gap: 4, marginTop: 4 }}
        >
          {artifacts.map((artifact) => (
            <div
              key={artifact.relative_path}
              style={{
                alignItems: "center",
                border: "1px solid var(--border)",
                borderRadius: 3,
                display: "grid",
                gap: 4,
                gridTemplateColumns: "minmax(0, 1fr) auto",
                padding: "3px 4px",
              }}
            >
              <span
                style={{
                  color: "var(--text-secondary)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 9,
                  minWidth: 0,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
                title={artifact.relative_path}
              >
                {artifact.relative_path}
              </span>
              <button
                aria-label={`Download project evidence artifact ${artifact.relative_path}`}
                disabled={loading}
                onClick={() => onDownloadArtifact(artifact.relative_path)}
                style={{
                  background: "var(--bg-tertiary)",
                  border: "1px solid var(--border)",
                  borderRadius: 3,
                  color: "var(--text-secondary)",
                  cursor: loading ? "wait" : "pointer",
                  fontSize: 9,
                  padding: "2px 5px",
                }}
                type="button"
              >
                Download
              </button>
              <small
                style={{
                  color: "var(--text-muted)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 8,
                  gridColumn: "1 / -1",
                }}
              >
                {formatArtifactSize(artifact.size_bytes)} - sha {artifact.sha256.slice(0, 12)}
              </small>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
