import { useState } from "react";
import { useStudioStore } from "../stores/studio";
import { formatCitation } from "../citation";
import EvidenceTierBadge from "./EvidenceTierBadge";

export default function ModelInfo() {
  const { sourceMode, modelDetail, equations, odeParams, odeInit, dt, duration } = useStudioStore();
  const [copied, setCopied] = useState(false);

  const citation = modelDetail ? formatCitation(modelDetail.provenance, modelDetail.name) : "";
  function copyCitation() {
    if (!citation) return;
    void navigator.clipboard?.writeText(citation);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  }

  const nSteps = Math.min(Math.floor(duration / dt), 100_000);

  if (sourceMode === "model" && modelDetail) {
    const prov = modelDetail.provenance;
    return (
      <div>
        <div style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4 }}>
          {modelDetail.docstring || modelDetail.name}
        </div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 4, alignItems: "center" }}>
          <EvidenceTierBadge tier={modelDetail.tier} evidenceKind={modelDetail.evidence_kind} full />
          {modelDetail.family && (
            <span style={{ fontSize: 10, color: "var(--accent)" }}>{modelDetail.family}</span>
          )}
          {modelDetail.maturity && (
            <span style={{
              fontSize: 8, padding: "0 4px", borderRadius: 2,
              background: "var(--bg-tertiary)", color: "var(--text-muted)", textTransform: "uppercase",
            }}>{modelDetail.maturity}</span>
          )}
        </div>
        {prov && (prov.doi || prov.authors.length > 0) && (
          <div style={{
            fontSize: 10, color: "var(--text-secondary)", marginBottom: 4,
            padding: "3px 5px", background: "var(--bg-tertiary)", borderRadius: "var(--radius)",
          }}>
            <div>
              {prov.authors.join(", ")}
              {prov.year ? ` (${prov.year})` : ""}
            </div>
            <div style={{ display: "flex", gap: 6, alignItems: "center", marginTop: 2 }}>
              {prov.doi && (
                <a href={`https://doi.org/${prov.doi}`} target="_blank" rel="noreferrer"
                  title="Open the cited paper"
                  style={{ color: "var(--accent)", textDecoration: "none", fontFamily: "var(--font-mono)" }}>
                  doi:{prov.doi}
                </a>
              )}
              {citation && (
                <button type="button" onClick={copyCitation} title={citation}
                  style={{
                    fontSize: 9, padding: "0 5px", cursor: "pointer",
                    background: "var(--bg-secondary)", color: "var(--text-secondary)",
                    border: "1px solid var(--border)", borderRadius: "var(--radius)",
                  }}>
                  {copied ? "✓ copied" : "⧉ How to cite"}
                </button>
              )}
            </div>
          </div>
        )}
        <div className="model-info">
          <div className="info-item">
            <span className="info-label">vars:</span>
            <span className="info-value">
              {modelDetail.state_vars.map((s) => s.name).join(", ")}
            </span>
          </div>
          <div className="info-item">
            <span className="info-label">params:</span>
            <span className="info-value">{modelDetail.params.length}</span>
          </div>
          <div className="info-item">
            <span className="info-label">dt:</span>
            <span className="info-value">{modelDetail.dt}</span>
          </div>
          <div className="info-item">
            <span className="info-label">steps:</span>
            <span className="info-value">{nSteps.toLocaleString()}</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="model-info">
      <div className="info-item">
        <span className="info-label">eqs:</span>
        <span className="info-value">{equations.length}</span>
      </div>
      <div className="info-item">
        <span className="info-label">vars:</span>
        <span className="info-value">{Object.keys(odeInit).join(", ")}</span>
      </div>
      <div className="info-item">
        <span className="info-label">params:</span>
        <span className="info-value">{Object.keys(odeParams).length}</span>
      </div>
      <div className="info-item">
        <span className="info-label">steps:</span>
        <span className="info-value">{nSteps.toLocaleString()}</span>
      </div>
    </div>
  );
}
