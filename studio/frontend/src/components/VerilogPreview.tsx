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
  const { verilogSrc, sourceMode } = useStudioStore();

  if (sourceMode !== "ode") {
    return (
      <div style={{
        flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
        color: "var(--text-muted)", fontSize: 13,
      }}>
        Verilog compilation is available in Custom ODE mode.
        <br />Switch to ODE mode and click Compile.
      </div>
    );
  }

  if (!verilogSrc) {
    return (
      <div style={{
        flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
        color: "var(--text-muted)", fontSize: 13,
      }}>
        Click "Compile" to generate Verilog RTL from your ODE.
      </div>
    );
  }

  return (
    <div style={{ flex: 1, padding: 8 }}>
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
  );
}
