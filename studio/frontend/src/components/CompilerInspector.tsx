// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Native compiler source, cadence and validation inspector

import type { CSSProperties } from "react";
import { useStudioStore } from "../stores/studio";
import { downloadBrowserArtefact } from "../browserArtefactDownload";
import EquationIrOptions from "./EquationIrOptions";
import EvidenceBundleArtifactList from "./EvidenceBundleArtifactList";

const buttonStyle: CSSProperties = {
  padding: "3px 8px", border: "1px solid var(--control-border)", borderRadius: "var(--radius)",
  background: "var(--bg-secondary)", color: "var(--text-primary)", cursor: "pointer",
  fontSize: "var(--fs-body)",
};
const stripStyle: CSSProperties = {
  padding: "6px 12px", fontSize: "var(--fs-body)", fontFamily: "var(--font-mono)",
  color: "var(--text-secondary)", background: "var(--bg-secondary)",
  borderBottom: "1px solid var(--border)", display: "flex", gap: 12,
  alignItems: "center", flexWrap: "wrap",
};
const codeStyle: CSSProperties = {
  flex: 1, margin: 0, padding: 8, overflow: "auto", fontSize: "var(--fs-body)",
  fontFamily: "var(--font-mono)", color: "var(--text-primary)",
  background: "var(--bg-primary)", whiteSpace: "pre-wrap",
};

/**
 * Inspect real graph facts, semantic replay, output cadence and generated source.
 *
 * @returns Compiler controls and source, with validation claims scoped to actual results.
 */
export default function CompilerInspector() {
  const {
    compileTraceability, compileEvidenceBundle, compileEvidenceBundleError,
    compileEvidenceBundleLoading, createEvidenceBundleForSurface,
    downloadEvidenceBundleArtifactForSurface, irText, svSource, irErrors, irResult,
    irRequestKey, isSimulating, verilogSrc, sourceMode, runBuildIR, runEmitSV, runVerifyIR, runReplayIR,
  } = useStudioStore();
  const rtlSource = svSource || verilogSrc;
  const moduleName = irResult?.graph_name ?? compileTraceability?.output.module_name ?? "neuron";

  let replay: Record<string, unknown> | null = null;
  let jobs: string[] = [];
  if (irResult !== null) {
    replay = {
      method: "POST", route: "/api/ir/replay", body: { ir_text: irResult.ir_text },
      note: "Recompile bound equation source and require identical native graph.",
    };
    jobs = [irResult.studio_job_receipt.job_id];
  } else if (compileTraceability !== null) {
    const source = compileTraceability.source_payload;
    const body = "equations" in source
      ? { ...source, module_name: compileTraceability.output.module_name }
      : { model_name: source.model_name, params: source.params, dt: source.dt,
        integrator: source.integrator, q_format: source.q_format,
        module_name: compileTraceability.output.module_name };
    replay = {
      method: "POST", route: "equations" in source ? "/api/compile" : "/api/models/compile", body,
      note: `compile trace ${compileTraceability.traceability_sha256}`,
      source_payload_sha256: compileTraceability.input_sha256,
    };
  }

  const inspectionDisabled = isSimulating || !irText || irRequestKey === null;
  return <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0, overflow: "hidden" }}>
    <EquationIrOptions />
    <div style={stripStyle} aria-label="Native equation compiler actions">
      <button type="button" style={buttonStyle} disabled={isSimulating || sourceMode !== "ode"}
        onClick={() => { void runBuildIR(); }}>Build IR</button>
      <button type="button" style={buttonStyle} disabled={isSimulating || sourceMode !== "ode"}
        onClick={() => { void runEmitSV(); }}>Emit SV</button>
      <button type="button" style={buttonStyle} disabled={inspectionDisabled}
        onClick={() => { void runVerifyIR(); }}>Check graph structure</button>
      <button type="button" style={buttonStyle} disabled={inspectionDisabled}
        onClick={() => { void runReplayIR(); }}>Replay source</button>
      {isSimulating && <span>Compiling…</span>}
    </div>
    {irResult !== null && <div style={stripStyle} aria-label="Native graph validation">
      <span>{irResult.valid ? "Graph structure valid" : "Graph structure invalid"}</span>
      <span>{irResult.source_replay === "passed" ? "Source replay passed"
        : irResult.source_replay === "unavailable" ? "Source replay unavailable" : "Source replay not run"}</span>
      <span>{irResult.n_ops} operations · {irResult.n_inputs} inputs · {irResult.n_outputs} outputs</span>
      {irResult.clocks_per_timestep !== null && <span>{irResult.clocks_per_timestep} clocks per timestep</span>}
      {irResult.pipeline_registers !== null && <span>{irResult.pipeline_registers} product registers</span>}
      {irResult.output_widths !== null && <span>Output widths: {irResult.output_widths.map(
        (port) => `${port.name}: ${port.width} bits`,
      ).join(", ")}</span>}
      {irResult.source_sha256 !== null && <span title={irResult.source_sha256}>source {irResult.source_sha256.slice(0, 12)}</span>}
      {irResult.source_graph_sha256 !== null && <span title={irResult.source_graph_sha256}>graph {irResult.source_graph_sha256.slice(0, 12)}</span>}
      <span>job {irResult.studio_job_receipt.job_id}</span>
    </div>}
    {compileTraceability !== null && <div style={stripStyle} aria-label="RTL source traceability">
      <span>schema {compileTraceability.schema_version}</span>
      <span>class {compileTraceability.evidence_classification}</span>
      <span>status {compileTraceability.status}</span>
      <span>module {compileTraceability.output.module_name}</span>
      <span>input {compileTraceability.input_sha256.slice(0, 12)}</span>
      <span>rtl {compileTraceability.output.rtl_sha256.slice(0, 12)}</span>
      <span>trace {compileTraceability.traceability_sha256.slice(0, 12)}</span>
    </div>}
    {(irText || rtlSource) && <div style={stripStyle}>
      {irResult === null && <span>Generated source; native graph validation unavailable.</span>}
      <button type="button" style={buttonStyle} disabled={!irText || isSimulating}
        onClick={() => { downloadBrowserArtefact(new Blob([irText], { type: "text/plain" }), moduleName + ".scir"); }}>Download IR</button>
      <button type="button" style={buttonStyle} disabled={!rtlSource || isSimulating}
        onClick={() => { downloadBrowserArtefact(new Blob([rtlSource], { type: "text/plain" }), moduleName + ".sv"); }}>Download SystemVerilog</button>
      <button type="button" style={buttonStyle} disabled={irRequestKey === null || isSimulating}
        onClick={irRequestKey === null ? undefined : () => {
          downloadBrowserArtefact(
            new Blob([irRequestKey], { type: "application/json" }), moduleName + ".request.json",
          );
        }}>Download compiler request</button>
      <button type="button" style={buttonStyle} aria-label="Export compile evidence bundle"
        disabled={compileEvidenceBundleLoading || isSimulating || irResult === null && compileTraceability === null}
        onClick={replay === null ? undefined : () => {
          void createEvidenceBundleForSurface("compile", {
            audit_limit: 100, analysis_results: [], command_replay: replay,
            default_flow_attestations: [], default_flow_runs: [], include_audit: true,
            job_ids: jobs, model_scan_results: [], project_name: null,
            simulation_results: [], weight_restore_results: [], weight_restore_attach_results: [],
          });
        }}>Export compile evidence bundle</button>
      {compileEvidenceBundle !== null && <span>bundle {compileEvidenceBundle.bundle_id}</span>}
      {compileEvidenceBundleError !== null && <span style={{ color: "#ff5252" }}>{compileEvidenceBundleError}</span>}
      {compileEvidenceBundle !== null && <EvidenceBundleArtifactList
        ariaLabel="Compile evidence bundle artifacts" artifacts={compileEvidenceBundle.artifacts}
        downloadLabelPrefix="Download compile evidence artifact" loading={compileEvidenceBundleLoading}
        onDownloadArtifact={(relativePath) => {
          void downloadEvidenceBundleArtifactForSurface("compile", relativePath);
        }} />}
    </div>}
    {irErrors.length > 0 && <div role="alert" style={{ ...stripStyle, color: "#ff5252", maxHeight: 80, overflowY: "auto" }}>
      {irErrors.map((error, index) => <div key={index}>{error}</div>)}
    </div>}
    {!irText && !rtlSource
      ? <div style={{ flex: 1, display: "grid", placeItems: "center", color: "var(--text-muted)", padding: 16 }}>
        <p>Build IR from the current equations, then replay its source or emit native SystemVerilog.
          Graph checks, source consistency, RTL tool checks and hardware validation are separate results.</p>
      </div>
      : <div style={{ flex: 1, display: "flex", minHeight: 0, overflow: "hidden" }}>
        <section style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, borderRight: "1px solid var(--border)" }}>
          <div style={stripStyle}>SC Intermediate Representation</div>
          <pre aria-label="Native intermediate representation" style={codeStyle}>{irText || "(no native IR generated)"}</pre>
        </section>
        <section style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
          <div style={stripStyle}>SystemVerilog</div>
          <pre aria-label="Generated SystemVerilog" style={codeStyle}>{rtlSource || "(use Emit SV to generate source)"}</pre>
        </section>
      </div>}
  </div>;
}
