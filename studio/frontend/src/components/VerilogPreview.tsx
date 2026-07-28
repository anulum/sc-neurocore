// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import Editor from "@monaco-editor/react";
import { useStudioStore } from "../stores/studio";

export default function VerilogPreview() {
  const {
    cosimResult,
    compileTraceability,
    isSimulating,
    modelDetail,
    modelIntegrator,
    runCosim,
    sourceMode,
    verilogSrc,
  } = useStudioStore();

  if (!verilogSrc) {
    return (
      <div style={{
        flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
        color: "var(--text-muted)", fontSize: 13,
      }}>
        Click "Compile" to generate Verilog RTL from your {sourceMode === "model" ? "selected model" : "ODE"}.
      </div>
    );
  }

  const cosimSupported = sourceMode === "model"
    && (modelDetail?.compile_configuration?.cosim_integrators ?? []).includes(modelIntegrator);
  const cosimMatchesCompile = cosimResult !== null
    && compileTraceability !== null
    && cosimResult.rtl.source_sha256 === compileTraceability.output.rtl_sha256;

  return (
    <div style={{ flex: 1, padding: 8, display: "flex", flexDirection: "column", overflow: "hidden" }}>
      {sourceMode === "model" && (
        <div
          aria-label="Selected model RTL co-simulation"
          style={{
            alignItems: "center",
            background: "var(--bg-secondary)",
            border: "1px solid var(--border)",
            display: "flex",
            flexWrap: "wrap",
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            gap: 10,
            marginBottom: 6,
            padding: "6px 8px",
          }}
        >
          <button
            aria-label="Run selected model RTL co-simulation"
            disabled={!cosimSupported || isSimulating}
            onClick={() => { void runCosim(); }}
            type="button"
          >
            {isSimulating ? "Running co-sim" : "Run RTL co-sim"}
          </button>
          {!cosimSupported && (
            <span style={{ color: "#c98a8a" }}>
              Integrator {modelIntegrator} has no bit-exact co-simulation path.
            </span>
          )}
          {cosimResult && (
            <>
              <strong style={{ color: cosimResult.bit_exact ? "#7bc67b" : "#ff5252" }}>
                {cosimResult.bit_exact ? "BIT-EXACT PASS" : "PARITY FAIL"}
              </strong>
              <span>{cosimResult.sample_count} cycles</span>
              <span>{cosimResult.configuration.q_format}</span>
              <span>{cosimResult.configuration.integrator}</span>
              <span>{cosimResult.signals.length} signals</span>
              <span>trace {cosimResult.rtl.trace_sha256.slice(0, 12)}</span>
              <span style={{ color: cosimMatchesCompile ? "#7bc67b" : "#ff5252" }}>
                {cosimMatchesCompile ? "compiled RTL match" : "compiled RTL mismatch"}
              </span>
              <span title={Object.values(cosimResult.tools).join(" | ")}>
                GCC + Icarus/VVP
              </span>
              {cosimResult.first_mismatch && (
                <span style={{ color: "#ff5252" }}>
                  cycle {cosimResult.first_mismatch.cycle}: {cosimResult.first_mismatch.signals.join(", ")}
                </span>
              )}
            </>
          )}
        </div>
      )}
      <div style={{ flex: 1, minHeight: 0 }}>
        <Editor
          height="100%"
        defaultLanguage="systemverilog"
        value={verilogSrc}
        options={{
          readOnly: true,
          minimap: { enabled: false },
          fontSize: 12,
          fontFamily: "var(--font-mono)",
          scrollBeyondLastLine: false,
          lineNumbers: "on",
          renderLineHighlight: "none",
        }}
        theme="vs-dark"
        />
      </div>
    </div>
  );
}
