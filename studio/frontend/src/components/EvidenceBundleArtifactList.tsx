import type { StudioJobArtifact } from "../api/client";

export interface EvidenceBundleArtifactListProps {
  ariaLabel: string;
  artifacts: StudioJobArtifact[];
  downloadLabelPrefix: string;
  loading: boolean;
  onDownloadArtifact: (relativePath: string) => void;
}

function formatArtifactSize(sizeBytes: number): string {
  if (sizeBytes < 1024) return `${sizeBytes} B`;
  if (sizeBytes < 1024 * 1024) return `${(sizeBytes / 1024).toFixed(1)} KiB`;
  return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MiB`;
}

export default function EvidenceBundleArtifactList({
  ariaLabel,
  artifacts,
  downloadLabelPrefix,
  loading,
  onDownloadArtifact,
}: EvidenceBundleArtifactListProps) {
  if (artifacts.length === 0) {
    return null;
  }

  return (
    <div
      aria-label={ariaLabel}
      style={{ display: "grid", flexBasis: "100%", gap: 4, marginTop: 4, width: "100%" }}
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
            aria-label={`${downloadLabelPrefix} ${artifact.relative_path}`}
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
  );
}
