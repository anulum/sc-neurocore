import { useStudioStore } from "../stores/studio";
import EvidenceBundleArtifactList from "./EvidenceBundleArtifactList";

export default function CompilerInspector() {
  const {
    compileTraceability,
    compileEvidenceBundle,
    compileEvidenceBundleError,
    compileEvidenceBundleLoading,
    createEvidenceBundleForSurface,
    downloadEvidenceBundleArtifactForSurface,
    irText,
    svSource,
    irErrors,
    isSimulating,
    verilogSrc,
  } = useStudioStore();
  const rtlSource = svSource || verilogSrc;

  function exportCompileEvidence() {
    if (compileTraceability === null) {
      return;
    }
    void createEvidenceBundleForSurface("compile", {
      audit_limit: 100,
      analysis_results: [],
      command_replay: {
        method: "POST",
        note: `compile trace ${compileTraceability.traceability_sha256}`,
        request_sha256: compileTraceability.input_sha256,
        route: "/api/ir/emit-sv-direct",
      },
      default_flow_attestations: [],
      default_flow_runs: [],
      include_audit: true,
      job_ids: [],
      model_scan_results: [],
      project_name: `compile-${compileTraceability.output.module_name}`,
      simulation_results: [],
      weight_restore_results: [],
      weight_restore_attach_results: [],
    });
  }

  if (!irText && !rtlSource) {
    return (
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)" }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 14, marginBottom: 8 }}>Compiler Inspector</div>
          <div style={{ fontSize: 11 }}>
            Click <strong>Build IR</strong> to compile your equation to stochastic computing IR,
            then <strong>Emit SV</strong> to generate SystemVerilog.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
      {/* Verification badge */}
      <div style={{
        padding: "4px 12px", fontSize: 10, fontFamily: "var(--font-mono)",
        background: irErrors.length > 0 ? "rgba(255,82,82,0.15)" : "rgba(129,199,132,0.15)",
        borderBottom: "1px solid var(--border)",
        color: irErrors.length > 0 ? "#ff5252" : "#81c784",
        display: "flex", gap: 12, alignItems: "center",
      }}>
        <span>{irErrors.length > 0 ? `${irErrors.length} error(s)` : "Verified"}</span>
        {isSimulating && <span style={{ color: "var(--text-muted)" }}>compiling...</span>}
      </div>

      {compileTraceability && (
        <div style={{
          padding: "6px 12px", fontSize: 10, fontFamily: "var(--font-mono)",
          color: "var(--text-secondary)", background: "var(--bg-secondary)",
          borderBottom: "1px solid var(--border)", display: "flex", gap: 14,
          alignItems: "center", flexWrap: "wrap",
        }}>
          <span>schema {compileTraceability.schema_version}</span>
          <span>class {compileTraceability.evidence_classification}</span>
          <span>status {compileTraceability.status}</span>
          <span>module {compileTraceability.output.module_name}</span>
          <span>input {compileTraceability.input_sha256.slice(0, 12)}</span>
          <span>rtl {compileTraceability.output.rtl_sha256.slice(0, 12)}</span>
          <span>trace {compileTraceability.traceability_sha256.slice(0, 12)}</span>
          <button
            aria-label="Export compile evidence bundle"
            disabled={compileEvidenceBundleLoading}
            onClick={exportCompileEvidence}
            style={{
              padding: "3px 8px",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
              background: "var(--accent)",
              color: "var(--bg-primary)",
              cursor: compileEvidenceBundleLoading ? "wait" : "pointer",
              fontSize: 10,
            }}
            type="button"
          >
            Export
          </button>
          {compileEvidenceBundle && (
            <span>bundle {compileEvidenceBundle.bundle_id}</span>
          )}
          {compileEvidenceBundleError && (
            <span style={{ color: "#ff5252" }}>{compileEvidenceBundleError}</span>
          )}
          {compileEvidenceBundle && (
            <EvidenceBundleArtifactList
              ariaLabel="Compile evidence bundle artifacts"
              artifacts={compileEvidenceBundle.artifacts}
              downloadLabelPrefix="Download compile evidence artifact"
              loading={compileEvidenceBundleLoading}
              onDownloadArtifact={(relativePath) => {
                void downloadEvidenceBundleArtifactForSurface("compile", relativePath);
              }}
            />
          )}
        </div>
      )}

      {/* Error list */}
      {irErrors.length > 0 && (
        <div style={{
          padding: "4px 12px", fontSize: 10, color: "#ff5252",
          background: "var(--bg-secondary)", borderBottom: "1px solid var(--border)",
          maxHeight: 60, overflowY: "auto",
        }}>
          {irErrors.map((err, i) => (
            <div key={i}>{err}</div>
          ))}
        </div>
      )}

      {/* Split pane: IR left, SV right */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        {/* IR panel */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", borderRight: "1px solid var(--border)" }}>
          <div style={{
            padding: "3px 8px", fontSize: 9, fontWeight: 700,
            color: "var(--accent)", background: "var(--bg-secondary)",
            borderBottom: "1px solid var(--border)", textTransform: "uppercase",
          }}>SC Intermediate Representation</div>
          <pre style={{
            flex: 1, margin: 0, padding: 8, overflow: "auto",
            fontSize: 11, fontFamily: "var(--font-mono)",
            color: "var(--text-primary)", background: "var(--bg-primary)",
            whiteSpace: "pre-wrap",
          }}>{irText || "(no IR generated)"}</pre>
        </div>

        {/* SystemVerilog panel */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
          <div style={{
            padding: "3px 8px", fontSize: 9, fontWeight: 700,
            color: "#a5d6a7", background: "var(--bg-secondary)",
            borderBottom: "1px solid var(--border)", textTransform: "uppercase",
          }}>SystemVerilog</div>
          <pre style={{
            flex: 1, margin: 0, padding: 8, overflow: "auto",
            fontSize: 11, fontFamily: "var(--font-mono)",
            color: "var(--text-primary)", background: "var(--bg-primary)",
            whiteSpace: "pre-wrap",
          }}>{rtlSource || "(click Emit SV to generate)"}</pre>
        </div>
      </div>
    </div>
  );
}
