import type { StudioJobArtifact } from "../api/client";
import type { ProjectEvidenceModel } from "../projectEvidence";
import EvidenceBundleArtifactList from "./EvidenceBundleArtifactList";
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
      <EvidenceBundleArtifactList
        ariaLabel="Project evidence bundle artifacts"
        artifacts={artifacts}
        downloadLabelPrefix="Download project evidence artifact"
        loading={loading}
        onDownloadArtifact={onDownloadArtifact}
      />
    </div>
  );
}
