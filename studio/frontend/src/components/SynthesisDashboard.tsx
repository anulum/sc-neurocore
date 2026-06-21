import { useEffect, useState } from "react";
import { useStudioStore } from "../stores/studio";
import type {
  SynthesisTargetProvenance,
  SynthesisTargetProvenanceMatrix,
} from "../api/client";
import SynthesisEvidenceControls from "./SynthesisEvidenceControls";

function ResourceBar({ label, used, total, color }: {
  label: string; used: number; total: number; color: string;
}) {
  const pct = total > 0 ? Math.min((used / total) * 100, 100) : 0;
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, marginBottom: 2 }}>
        <span style={{ color: "var(--text-secondary)" }}>{label}</span>
        <span style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
          {used} / {total} ({pct.toFixed(1)}%)
        </span>
      </div>
      <div style={{
        height: 8, background: "var(--bg-tertiary)", borderRadius: 4, overflow: "hidden",
      }}>
        <div style={{
          height: "100%", width: `${pct}%`, background: color,
          borderRadius: 4, transition: "width 0.3s",
        }} />
      </div>
    </div>
  );
}

function TargetComparisonRow({ target, result }: {
  target: string; result: { success: boolean; error?: string; resources?: { luts: number; ffs: number; brams: number; dsps: number }; capacity?: { luts: number; ffs: number; brams: number; dsps: number }; utilisation?: Record<string, number> };
}) {
  if (!result.success) {
    return (
      <tr>
        <td style={{ padding: "3px 8px", fontWeight: 600 }}>{target.toUpperCase()}</td>
        <td colSpan={4} style={{ padding: "3px 8px", color: "#ff5252", fontSize: 10 }}>
          {result.error?.slice(0, 60) || "Failed"}
        </td>
      </tr>
    );
  }
  const r = result.resources!;
  const u = result.utilisation!;
  return (
    <tr>
      <td style={{ padding: "3px 8px", fontWeight: 600 }}>{target.toUpperCase()}</td>
      <td style={{ padding: "3px 8px", fontFamily: "var(--font-mono)" }}>{r.luts} ({u.luts}%)</td>
      <td style={{ padding: "3px 8px", fontFamily: "var(--font-mono)" }}>{r.ffs} ({u.ffs}%)</td>
      <td style={{ padding: "3px 8px", fontFamily: "var(--font-mono)" }}>{r.brams} ({u.brams}%)</td>
      <td style={{ padding: "3px 8px", fontFamily: "var(--font-mono)" }}>{r.dsps} ({u.dsps}%)</td>
    </tr>
  );
}

function ProvenanceSummary({ provenance }: { provenance: SynthesisTargetProvenance }) {
  const synthesisTool = provenance.tools.find((tool) => tool.role === "synthesis");
  const pnrTool = provenance.tools.find((tool) => tool.role === "place_and_route");
  return (
    <div style={{
      marginTop: 10, padding: 8, background: "var(--bg-secondary)",
      borderRadius: 4, fontSize: 10, color: "var(--text-secondary)",
    }}>
      <div style={{ fontWeight: 600, marginBottom: 4 }}>Target provenance</div>
      <div>Command: {provenance.synthesis_command}</div>
      <div>
        Synthesis tool: {synthesisTool?.executable ?? "yosys"} (
        {provenance.synthesis_ready ? "available" : "missing"}
        {synthesisTool?.version ? `, ${synthesisTool.version}` : ""})
      </div>
      <div>
        PnR: {pnrTool?.executable ?? "not configured"} (
        {provenance.pnr_tool ? (provenance.pnr_ready ? "available" : "missing") : "not required"})
      </div>
      <div>Evidence: {provenance.evidence_classification}</div>
      <div>Status: {provenance.status}</div>
    </div>
  );
}

function readinessLabel(isReady: boolean): "ready" | "missing" {
  return isReady ? "ready" : "missing";
}

function toolLabel(provenance: SynthesisTargetProvenance, role: string): string {
  const tool = provenance.tools.find((item) => item.role === role);
  if (tool === undefined) {
    return role === "place_and_route" && provenance.pnr_tool === null ? "not required" : "missing";
  }
  const version = tool.version === null ? "" : ` ${tool.version}`;
  return `${tool.executable} ${readinessLabel(tool.available)}${version}`;
}

export function ProvenanceMatrixSummary({
  matrix,
}: {
  matrix: SynthesisTargetProvenanceMatrix;
}) {
  const entries = Object.entries(matrix.targets).sort(([left], [right]) =>
    left.localeCompare(right),
  );
  return (
    <div style={{
      marginTop: 10, padding: 8, background: "var(--bg-secondary)",
      borderRadius: 4, fontSize: 10, color: "var(--text-secondary)",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 8, marginBottom: 6 }}>
        <strong>Target provenance matrix</strong>
        <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
          {matrix.evidence_classification} / {matrix.status} / {matrix.matrix_sha256.slice(0, 12)}
        </span>
      </div>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ borderBottom: "1px solid var(--border)" }}>
            <th style={{ padding: "3px 6px", textAlign: "left" }}>Target</th>
            <th style={{ padding: "3px 6px", textAlign: "left" }}>Device</th>
            <th style={{ padding: "3px 6px", textAlign: "left" }}>Synthesis</th>
            <th style={{ padding: "3px 6px", textAlign: "left" }}>PnR</th>
            <th style={{ padding: "3px 6px", textAlign: "left" }}>Evidence</th>
            <th style={{ padding: "3px 6px", textAlign: "left" }}>Status</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([target, provenance]) => (
            <tr key={target} style={{ borderBottom: "1px solid var(--border)" }}>
              <td style={{ padding: "3px 6px", fontWeight: 600 }}>{target.toUpperCase()}</td>
              <td style={{ padding: "3px 6px" }}>{provenance.device ?? "none"}</td>
              <td style={{ padding: "3px 6px" }}>
                {readinessLabel(provenance.synthesis_ready)} - {toolLabel(provenance, "synthesis")}
              </td>
              <td style={{ padding: "3px 6px" }}>
                {readinessLabel(provenance.pnr_ready)} - {toolLabel(provenance, "place_and_route")}
              </td>
              <td style={{ padding: "3px 6px" }}>{provenance.evidence_classification}</td>
              <td style={{ padding: "3px 6px" }}>{provenance.status}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function SynthesisDashboard() {
  const {
    synthResult, synthEstimate, multiTargetResult,
    synthesisEvidenceBundle, synthesisEvidenceBundleError, synthesisEvidenceBundleLoading,
    latestSynthesisJobId, latestMultiTargetSynthesisJobId,
    synthTarget, toolsAvailable, svSource, verilogSrc,
    irText,
    createEvidenceBundleForSurface, downloadEvidenceBundleArtifactForSurface,
    setSynthTarget, runSynthesis, runMultiTargetSynthesis, runSynthEstimate,
    checkSynthTools, isSimulating,
  } = useStudioStore();
  const [synthesisBundleRequested, setSynthesisBundleRequested] = useState(false);

  useEffect(() => { checkSynthTools(); }, [checkSynthTools]);

  const targets = ["ice40", "ecp5", "gowin", "xilinx"];
  const hasSV = svSource.length > 0 || verilogSrc.length > 0;
  const hasIR = irText.length > 0;
  const activeSynthesisJobId = multiTargetResult
    ? latestMultiTargetSynthesisJobId
    : latestSynthesisJobId;

  function exportSynthesisEvidence(projectName: string) {
    if (activeSynthesisJobId === null) {
      return;
    }
    setSynthesisBundleRequested(true);
    void createEvidenceBundleForSurface("synthesis", {
      audit_limit: 100,
      analysis_results: [],
      command_replay: null,
      default_flow_attestations: [],
      default_flow_runs: [],
      include_audit: true,
      job_ids: [activeSynthesisJobId],
      model_scan_results: [],
      project_name: projectName,
      simulation_results: [],
      weight_restore_results: [],
      weight_restore_attach_results: [],
    });
  }

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "auto" }}>
      {/* Header */}
      <div style={{
        padding: "8px 12px", background: "var(--bg-secondary)",
        borderBottom: "1px solid var(--border)",
        display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap",
      }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text-primary)" }}>
          FPGA Synthesis
        </span>
        <select
          value={synthTarget}
          onChange={(e) => setSynthTarget(e.target.value)}
          style={{ fontSize: 10, padding: "2px 6px" }}
        >
          {targets.map((t) => (
            <option key={t} value={t}>{t.toUpperCase()}</option>
          ))}
        </select>
        <button
          className="btn-simulate"
          onClick={runSynthesis}
          disabled={isSimulating || !hasSV}
          style={{
            background: "#a5d6a7", color: "#0d1117", border: "none",
            padding: "3px 10px", fontSize: 10,
          }}
        >
          {isSimulating ? "..." : "Synthesise"}
        </button>
        <button
          className="btn-simulate"
          onClick={runMultiTargetSynthesis}
          disabled={isSimulating || !hasSV}
          style={{
            background: "#80cbc4", color: "#0d1117", border: "none",
            padding: "3px 10px", fontSize: 10,
          }}
        >
          All Targets
        </button>
        {hasIR && (
          <button
            className="btn-simulate"
            onClick={runSynthEstimate}
            disabled={isSimulating}
            style={{
              background: "#ffcc80", color: "#0d1117", border: "none",
              padding: "3px 10px", fontSize: 10,
            }}
          >
            Estimate
          </button>
        )}
        {!hasSV && (
          <span style={{ fontSize: 9, color: "var(--text-muted)" }}>
            Generate Verilog first (RTL or SV button)
          </span>
        )}
      </div>

      {/* Tool status */}
      {toolsAvailable && (
        <div style={{
          padding: "4px 12px", fontSize: 9, display: "flex", gap: 12,
          borderBottom: "1px solid var(--border)", color: "var(--text-muted)",
        }}>
          {Object.entries(toolsAvailable).map(([name, info]) => (
            <span key={name} style={{ display: "flex", alignItems: "center", gap: 3 }}>
              <span style={{
                width: 6, height: 6, borderRadius: "50%",
                background: info.available ? "#81c784" : "#616161",
              }} />
              {name}
              {info.version && (
                <span style={{ fontSize: 8, color: "var(--text-muted)" }}> ({info.version})</span>
              )}
            </span>
          ))}
        </div>
      )}

      <div style={{ padding: 12, flex: 1, overflow: "auto" }}>
        {/* Estimate preview */}
        {synthEstimate && !synthResult && !multiTargetResult && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: "#ffcc80", marginBottom: 8 }}>
              {synthEstimate.target.toUpperCase()} — Resource Estimate (heuristic, no Yosys)
            </div>
            <ResourceBar
              label="LUTs" used={synthEstimate.resources.luts}
              total={synthEstimate.capacity.luts} color="rgba(79, 195, 247, 0.5)"
            />
            <ResourceBar
              label="Flip-Flops" used={synthEstimate.resources.ffs}
              total={synthEstimate.capacity.ffs} color="rgba(129, 199, 132, 0.5)"
            />
            <ResourceBar
              label="DSPs" used={synthEstimate.resources.dsps}
              total={synthEstimate.capacity.dsps} color="rgba(206, 147, 216, 0.5)"
            />
            <div style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 4 }}>
              Heuristic estimate from IR operation count. Run Yosys for exact numbers.
            </div>
          </div>
        )}

        {/* Multi-target comparison table */}
        {multiTargetResult && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: "#80cbc4", marginBottom: 8 }}>
              Multi-Target Comparison
            </div>
            <table style={{
              width: "100%", fontSize: 10, borderCollapse: "collapse",
              color: "var(--text-secondary)",
            }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <th style={{ padding: "3px 8px", textAlign: "left" }}>Target</th>
                  <th style={{ padding: "3px 8px", textAlign: "left" }}>LUTs</th>
                  <th style={{ padding: "3px 8px", textAlign: "left" }}>FFs</th>
                  <th style={{ padding: "3px 8px", textAlign: "left" }}>BRAMs</th>
                  <th style={{ padding: "3px 8px", textAlign: "left" }}>DSPs</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(multiTargetResult.targets).map(([target, result]) => (
                  <TargetComparisonRow key={target} target={target} result={result} />
                ))}
              </tbody>
            </table>
            <ProvenanceMatrixSummary matrix={multiTargetResult.target_provenance_matrix} />
            <SynthesisEvidenceControls
              bundle={synthesisBundleRequested ? synthesisEvidenceBundle : null}
              error={synthesisBundleRequested ? synthesisEvidenceBundleError : null}
              jobId={latestMultiTargetSynthesisJobId}
              loading={synthesisEvidenceBundleLoading}
              onDownloadArtifact={(relativePath) => {
                void downloadEvidenceBundleArtifactForSurface("synthesis", relativePath);
              }}
              onExport={() => exportSynthesisEvidence("synthesis-multi-target")}
            />
          </div>
        )}

        {/* Single-target synthesis result */}
        {synthResult && (
          <div>
            {!synthResult.success ? (
              <div style={{
                padding: 12, background: "rgba(255,82,82,0.1)", borderRadius: 4,
                color: "#ff5252", fontSize: 11,
              }}>
                {synthResult.error}
              </div>
            ) : (
              <>
                <div style={{ fontSize: 11, fontWeight: 600, color: "var(--accent)", marginBottom: 12 }}>
                  {synthResult.target.toUpperCase()} — Synthesis Results
                </div>

                <ResourceBar
                  label="LUTs" used={synthResult.resources.luts}
                  total={synthResult.capacity.luts} color="#4fc3f7"
                />
                <ResourceBar
                  label="Flip-Flops" used={synthResult.resources.ffs}
                  total={synthResult.capacity.ffs} color="#81c784"
                />
                <ResourceBar
                  label="Block RAMs" used={synthResult.resources.brams}
                  total={synthResult.capacity.brams} color="#ffb74d"
                />
                <ResourceBar
                  label="DSPs" used={synthResult.resources.dsps}
                  total={synthResult.capacity.dsps} color="#ce93d8"
                />

                <div style={{
                  marginTop: 12, padding: 8, background: "var(--bg-secondary)",
                  borderRadius: 4, fontSize: 10, fontFamily: "var(--font-mono)",
                  color: "var(--text-secondary)",
                }}>
                  <div>Cells: {synthResult.resources.cells}</div>
                  <div>Wires: {synthResult.resources.wires}</div>
                  {synthResult.log_excerpt && (
                    <div style={{ marginTop: 6, color: "var(--text-muted)", fontSize: 9, whiteSpace: "pre-wrap" }}>
                      {synthResult.log_excerpt}
                    </div>
                  )}
                </div>
                <ProvenanceSummary provenance={synthResult.target_provenance} />
              </>
            )}
            <SynthesisEvidenceControls
              bundle={synthesisBundleRequested ? synthesisEvidenceBundle : null}
              error={synthesisBundleRequested ? synthesisEvidenceBundleError : null}
              jobId={latestSynthesisJobId}
              loading={synthesisEvidenceBundleLoading}
              onDownloadArtifact={(relativePath) => {
                void downloadEvidenceBundleArtifactForSurface("synthesis", relativePath);
              }}
              onExport={() => exportSynthesisEvidence(`synthesis-${synthResult.target}`)}
            />
          </div>
        )}

        {/* Empty state */}
        {!synthResult && !multiTargetResult && !synthEstimate && hasSV && (
          <div style={{
            flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
            color: "var(--text-muted)", fontSize: 11, minHeight: 100,
          }}>
            Click Synthesise to run Yosys, or Estimate for a quick heuristic
          </div>
        )}
      </div>
    </div>
  );
}
