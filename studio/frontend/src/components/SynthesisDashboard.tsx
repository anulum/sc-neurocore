// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { useEffect } from "react";
import { useStudioStore } from "../stores/studio";
import type {
  SiliconTerminalResult,
  SynthesisTargetProvenance,
  SynthesisTargetProvenanceMatrix,
  SynthResult,
} from "../api/client";
import SynthesisEvidenceControls from "./SynthesisEvidenceControls";

/**
 * One resource's usage against the device's capacity, as a labelled bar.
 *
 * @param props - The resource, what it used, what the device holds, and the
 *   colour to draw it in.
 * @returns The bar.
 */
export function ResourceBar({ label, used, total, color }: {
  label: string; used: number; total: number; color: string;
}) {
  // The figure is the real share; only the bar stops at full. The text was
  // capped too, so 6237 of 5280 LUTs read "(100.0%)".
  const share = total > 0 ? (used / total) * 100 : null;
  const over = used > total;
  const pct = share === null ? (used > 0 ? 100 : 0) : Math.min(share, 100);
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-body)", marginBottom: 2 }}>
        <span style={{ color: "var(--text-secondary)" }}>{label}</span>
        <span style={{ color: over ? "var(--error)" : "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
          {used} / {total} ({share === null ? (used > 0 ? "device has none" : "none used") : `${share.toFixed(1)}%`})
          {over ? " over capacity" : ""}
        </span>
      </div>
      <div style={{
        height: 8, background: "var(--bg-tertiary)", borderRadius: 4, overflow: "hidden",
      }}>
        <div style={{
          height: "100%", width: `${pct}%`, background: over ? "var(--error)" : color,
          borderRadius: 4, transition: "width 0.3s",
        }} />
      </div>
    </div>
  );
}

/** The server's fit verdict, as a synthesis or an estimate carries it. */
export type FitVerdictFields = Pick<
  SynthResult, "target" | "fits_device" | "exceeds_capacity" | "capacity_device" | "uncounted_cells"
>;

/**
 * Say whether the device holds the design, beside its resource bars.
 *
 * The device is named: Gowin and Xilinx synthesis is not bound to one, so the
 * family alone would not say what the counts were judged against. A fit is by
 * count only, and a design with cells of unknown cost is not judged at all.
 *
 * @param props - The verdict, and whether it judges an estimate.
 * @returns The sentence, or nothing without a verdict.
 */
export function FitVerdict({ verdict, estimate = false }: {
  verdict: FitVerdictFields;
  estimate?: boolean;
}) {
  const fits = verdict.fits_device;
  if (fits === undefined) return null;
  const device = verdict.capacity_device ?? `${verdict.target.toUpperCase()} device`;
  const names: Record<string, string> = { luts: "LUTs", ffs: "flip-flops", brams: "block RAMs", dsps: "DSP blocks" };
  let text: string;
  if (fits === null) {
    const uncounted = Object.entries(verdict.uncounted_cells ?? {});
    const total = uncounted.reduce((sum, [, count]) => sum + count, 0);
    const cells = uncounted
      .map(([type, count]) => `${String(count)} ${type} ${count === 1 ? "cell" : "cells"}`)
      .join(", ");
    text = `Not judged against the ${device}: ${cells} ${total === 1 ? "has" : "have"} no counted cost, so these counts are a floor.`;
  } else if (fits) {
    text = estimate
      ? `The estimate fits the ${device}; synthesis counts the real design.`
      : `Fits the ${device} by count; placement and routing decide the rest.`;
  } else {
    const lacks = Object.entries(verdict.exceeds_capacity ?? {})
      .map(([key, row]) => `${String(row.needed)} ${names[key] ?? key} needed, ${String(row.available)} available`)
      .join("; ");
    text = estimate
      ? `The estimate does not fit the ${device}: ${lacks}.`
      : `Does not fit the ${device}: ${lacks}. Synthesis produced a netlist the device cannot hold.`;
  }
  const color = fits === null ? "var(--warning)" : fits ? "var(--success)" : "var(--error)";
  return (
    <p role="status" data-testid="synthesis-fit-verdict" style={{
      margin: "0 0 10px", fontSize: "var(--fs-body)", color,
    }}>{text}</p>
  );
}

/**
 * The message shown for a target that failed to synthesise.
 *
 * `||` and not `??`: an empty message is no message, and the fallback is what
 * the reader needs to see.
 *
 * @param error - The server's message, if it sent one.
 * @returns The text to show.
 */
function failureText(error: string | undefined): string {
  // eslint-disable-next-line @typescript-eslint/prefer-nullish-coalescing
  return error?.slice(0, 60) || "Failed";
}

/**
 * One target's row in the multi-target comparison.
 *
 * A failed target shows its message across the resource columns rather than
 * empty cells, so a failure is not read as a design that used nothing.
 *
 * @param props - The target and its result.
 * @returns The row.
 */
function TargetComparisonRow({ target, result }: {
  target: string; result: SynthResult;
}) {
  if (!result.success) {
    return (
      <tr>
        <td style={{ padding: "3px 8px", fontWeight: 600 }}>{target.toUpperCase()}</td>
        <td colSpan={4} style={{ padding: "3px 8px", color: "#ff5252", fontSize: "var(--fs-body)" }}>
          {failureText(result.error)}
        </td>
      </tr>
    );
  }
  const r = result.resources;
  const u = result.utilisation;
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

/**
 * What a target's figures rest on, stated beside them.
 *
 * @param props - The target's provenance.
 * @returns The summary.
 */
function ProvenanceSummary({ provenance }: { provenance: SynthesisTargetProvenance }) {
  const synthesisTool = provenance.tools.find((tool) => tool.role === "synthesis");
  const pnrTool = provenance.tools.find((tool) => tool.role === "place_and_route");
  return (
    <div style={{
      marginTop: 10, padding: 8, background: "var(--bg-secondary)",
      borderRadius: 4, fontSize: "var(--fs-body)", color: "var(--text-secondary)",
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

/**
 * The end of the pipeline: the routed design and the chain that produced it.
 *
 * @param props - The terminal result.
 * @returns The summary.
 */
export function SiliconTerminalSummary({ terminal }: { terminal: SiliconTerminalResult }) {
  return (
    <div style={{
      marginTop: 10, padding: 8, background: "var(--bg-secondary)",
      borderRadius: 4, fontSize: "var(--fs-body)", color: "var(--text-secondary)",
    }}>
      <div style={{ fontWeight: 600, marginBottom: 4 }}>
        Selected RTL synthesis/PnR terminal
      </div>
      <div>Status: {terminal.status}</div>
      <div>Model: {terminal.source_chain.model_name} / module {terminal.source_chain.module_name}</div>
      <div>RTL: {terminal.source_chain.rtl_sha256.slice(0, 12)}</div>
      <div>Netlist: {terminal.artifacts.netlist_sha256?.slice(0, 12) ?? "not produced"}</div>
      <div>
        Routed design: {terminal.artifacts.routed_design_sha256?.slice(0, 12) ?? "not produced"}
      </div>
      <div>Max frequency: {terminal.place_and_route?.max_freq_mhz ?? "unreported"} MHz</div>
      {!terminal.success && (
        <div style={{ color: "#ff5252", marginTop: 4 }}>
          {terminal.place_and_route?.error ?? "Terminal did not complete."}
        </div>
      )}
    </div>
  );
}

/**
 * The word shown for a readiness flag.
 *
 * @param isReady - Whether the step can run.
 * @returns The label.
 */
function readinessLabel(isReady: boolean): "ready" | "missing" {
  return isReady ? "ready" : "missing";
}

/**
 * Name the tool filling one role for a target, with its version.
 *
 * @param provenance - The target's provenance.
 * @param role - The role to look for.
 * @returns The tool and version, or a statement that there is none.
 */
function toolLabel(provenance: SynthesisTargetProvenance, role: string): string {
  const tool = provenance.tools.find((item) => item.role === role);
  if (tool === undefined) {
    return role === "place_and_route" && provenance.pnr_tool === null ? "not required" : "missing";
  }
  const version = tool.version === null ? "" : ` ${tool.version}`;
  return `${tool.executable} ${readinessLabel(tool.available)}${version}`;
}

/**
 * Every target's provenance at once, for comparing across devices.
 *
 * @param props - The matrix.
 * @returns The summary.
 */
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
      borderRadius: 4, fontSize: "var(--fs-body)", color: "var(--text-secondary)",
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

/**
 * The synthesis panel: tool availability, runs, targets and their evidence.
 *
 * @returns The panel.
 */
export default function SynthesisDashboard() {
  const {
    synthResult, synthEstimate, multiTargetResult,
    synthesisEvidenceBundle, synthesisEvidenceBundleError, synthesisEvidenceBundleLoading,
    latestSynthesisJobId, latestMultiTargetSynthesisJobId,
    synthTarget, toolsAvailable, svSource, verilogSrc,
    sourceMode, compileTraceability, cosimResult,
    irText,
    createEvidenceBundleForSurface, downloadEvidenceBundleArtifactForSurface,
    setSynthTarget, runSynthesis, runMultiTargetSynthesis, runSynthEstimate,
    checkSynthTools, isSimulating,
  } = useStudioStore();

  useEffect(() => { void checkSynthTools(); }, [checkSynthTools]);

  const targets = ["ice40", "ecp5", "gowin", "xilinx"];
  const hasSV = svSource.length > 0 || verilogSrc.length > 0;
  const hasIR = irText.length > 0;
  const selectedTerminalTarget = synthTarget === "ice40" || synthTarget === "ecp5";
  const selectedTerminalReady = sourceMode !== "model" || (
    selectedTerminalTarget
    && cosimResult?.bit_exact === true
    && compileTraceability !== null
    && cosimResult.rtl.source_sha256 === compileTraceability.output.rtl_sha256
  );
  const activeSynthesisJobId = multiTargetResult
    ? latestMultiTargetSynthesisJobId
    : latestSynthesisJobId;

  /** Gather this run's artefacts into an evidence bundle. */
  function exportSynthesisEvidence() {
    if (activeSynthesisJobId === null) {
      return;
    }
    void createEvidenceBundleForSurface("synthesis", {
      audit_limit: 100,
      analysis_results: [],
      command_replay: null,
      default_flow_attestations: [],
      default_flow_runs: [],
      include_audit: true,
      job_ids: [activeSynthesisJobId],
      model_scan_results: [],
      project_name: null,
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
        <h2 className="panel-header" style={{ margin: 0 }}>FPGA synthesis</h2>
        <select
          aria-label="FPGA target"
          value={synthTarget}
          onChange={(e) => { setSynthTarget(e.target.value); }}
          style={{ fontSize: "var(--fs-body)", padding: "2px 6px" }}
        >
          {targets.map((t) => (
            <option
              key={t}
              value={t}
              disabled={sourceMode === "model" && t !== "ice40" && t !== "ecp5"}
            >
              {t.toUpperCase()}
            </option>
          ))}
        </select>
        <button
          className="btn-simulate"
          onClick={() => { void runSynthesis(); }}
          disabled={isSimulating || !hasSV || !selectedTerminalReady}
          style={{
            background: "#a5d6a7", color: "#0d1117", border: "none",
            padding: "3px 10px", fontSize: "var(--fs-body)",
          }}
        >
          {isSimulating ? "..." : sourceMode === "model" ? "Synthesise + Route" : "Synthesise"}
        </button>
        <button
          className="btn-simulate"
          onClick={() => { void runMultiTargetSynthesis(); }}
          disabled={isSimulating || !hasSV || sourceMode === "model"}
          style={{
            background: "#80cbc4", color: "#0d1117", border: "none",
            padding: "3px 10px", fontSize: "var(--fs-body)",
          }}
        >
          All Targets
        </button>
        {hasIR && (
          <button
            className="btn-simulate"
            onClick={() => { void runSynthEstimate(); }}
            disabled={isSimulating}
            style={{
              background: "#ffcc80", color: "#0d1117", border: "none",
              padding: "3px 10px", fontSize: "var(--fs-body)",
            }}
          >
            Estimate
          </button>
        )}
        {!hasSV && (
          <span style={{ fontSize: "var(--fs-meta)", color: "var(--text-muted)" }}>
            Generate Verilog first: {sourceMode === "model" ? "\u201cGenerate RTL\u201d" : "\u201cGenerate RTL\u201d or \u201cEmit SystemVerilog\u201d"} in the header
          </span>
        )}
        {hasSV && sourceMode === "model" && !selectedTerminalReady && (
          <span style={{ fontSize: "var(--fs-meta)", color: "var(--text-muted)" }}>
            Compile and bit-exact co-simulate this RTL, then choose ICE40 or ECP5
          </span>
        )}
      </div>

      {/* Tool status */}
      {toolsAvailable && (
        <div style={{
          padding: "4px 12px", fontSize: "var(--fs-meta)", display: "flex", gap: 12,
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
                <span style={{ fontSize: "var(--fs-meta)", color: "var(--text-muted)" }}> ({info.version})</span>
              )}
            </span>
          ))}
        </div>
      )}

      <div style={{ padding: 12, flex: 1, overflow: "auto" }}>
        {/* Estimate preview */}
        {synthEstimate && !synthResult && !multiTargetResult && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: "var(--fs-body)", fontWeight: 600, color: "#ffcc80", marginBottom: 8 }}>
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
            <div style={{ fontSize: "var(--fs-meta)", color: "var(--text-muted)", marginTop: 4 }}>
              Heuristic estimate from IR operation count. Run Yosys for exact numbers.
            </div>
            <FitVerdict verdict={synthEstimate} estimate />
          </div>
        )}

        {/* Multi-target comparison table */}
        {multiTargetResult && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: "var(--fs-body)", fontWeight: 600, color: "#80cbc4", marginBottom: 8 }}>
              Multi-Target Comparison
            </div>
            <table style={{
              width: "100%", fontSize: "var(--fs-body)", borderCollapse: "collapse",
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
              bundle={synthesisEvidenceBundle}
              error={synthesisEvidenceBundleError}
              jobId={latestMultiTargetSynthesisJobId}
              loading={synthesisEvidenceBundleLoading}
              onDownloadArtifact={(relativePath) => {
                void downloadEvidenceBundleArtifactForSurface("synthesis", relativePath);
              }}
              onExport={exportSynthesisEvidence}
            />
          </div>
        )}

        {/* Single-target synthesis result */}
        {synthResult && (
          <div>
            {!synthResult.success ? (
              <div style={{
                padding: 12, background: "rgba(255,82,82,0.1)", borderRadius: 4,
                color: "#ff5252", fontSize: "var(--fs-body)",
              }}>
                {synthResult.error}
              </div>
            ) : (
              <>
                <div style={{ fontSize: "var(--fs-body)", fontWeight: 600, color: "var(--accent)", marginBottom: 12 }}>
                  {synthResult.target.toUpperCase()} — Synthesis Results
                </div>
                <FitVerdict verdict={synthResult} />

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
                  borderRadius: 4, fontSize: "var(--fs-body)", fontFamily: "var(--font-mono)",
                  color: "var(--text-secondary)",
                }}>
                  <div>Cells: {synthResult.resources.cells}</div>
                  <div>Wires: {synthResult.resources.wires}</div>
                  {synthResult.log_excerpt && (
                    <div style={{ marginTop: 6, color: "var(--text-muted)", fontSize: "var(--fs-meta)", whiteSpace: "pre-wrap" }}>
                      {synthResult.log_excerpt}
                    </div>
                  )}
                </div>
                <ProvenanceSummary provenance={synthResult.target_provenance} />
                {synthResult.silicon_terminal && (
                  <SiliconTerminalSummary terminal={synthResult.silicon_terminal} />
                )}
              </>
            )}
            <SynthesisEvidenceControls
              bundle={synthesisEvidenceBundle}
              error={synthesisEvidenceBundleError}
              jobId={latestSynthesisJobId}
              loading={synthesisEvidenceBundleLoading}
              onDownloadArtifact={(relativePath) => {
                void downloadEvidenceBundleArtifactForSurface("synthesis", relativePath);
              }}
              onExport={exportSynthesisEvidence}
            />
          </div>
        )}

        {/* Empty state */}
        {!synthResult && !multiTargetResult && !synthEstimate && hasSV && (
          <div style={{
            flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
            color: "var(--text-muted)", fontSize: "var(--fs-body)", minHeight: 100,
          }}>
            Click Synthesise to run Yosys, or Estimate for a quick heuristic
          </div>
        )}
      </div>
    </div>
  );
}
