// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio synthesis evidence controls
import type { StudioEvidenceBundleResponse } from "../api/client";
import EvidenceBundleArtifactList from "./EvidenceBundleArtifactList";

export interface SynthesisEvidenceControlsProps {
  bundle: StudioEvidenceBundleResponse | null;
  error: string | null;
  jobId: string | null;
  loading: boolean;
  onDownloadArtifact: (relativePath: string) => void;
  onExport: () => void;
}

export default function SynthesisEvidenceControls({
  bundle,
  error,
  jobId,
  loading,
  onDownloadArtifact,
  onExport,
}: SynthesisEvidenceControlsProps) {
  return (
    <div style={{
      marginTop: 10,
      padding: 8,
      background: "var(--bg-secondary)",
      borderRadius: 4,
      fontSize: 10,
      color: "var(--text-secondary)",
      display: "flex",
      gap: 8,
      alignItems: "center",
      flexWrap: "wrap",
    }}>
      <span style={{ fontWeight: 600 }}>Synthesis evidence</span>
      <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
        job {jobId ?? "pending"}
      </span>
      <button
        aria-label="Export synthesis evidence bundle"
        disabled={loading || jobId === null}
        onClick={onExport}
        style={{
          padding: "3px 8px",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          background: "var(--accent)",
          color: "var(--bg-primary)",
          cursor: loading || jobId === null ? "not-allowed" : "pointer",
          fontSize: 10,
        }}
        type="button"
      >
        Export
      </button>
      {bundle && <span>bundle {bundle.bundle_id}</span>}
      {error && <span style={{ color: "#ff5252" }}>{error}</span>}
      {bundle && (
        <EvidenceBundleArtifactList
          ariaLabel="Synthesis evidence bundle artefacts"
          artifacts={bundle.artifacts}
          downloadLabelPrefix="Download synthesis evidence artefact"
          loading={loading}
          onDownloadArtifact={onDownloadArtifact}
        />
      )}
    </div>
  );
}
