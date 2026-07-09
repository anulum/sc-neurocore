// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import Editor, { type Monaco } from "@monaco-editor/react";
import { useStudioStore } from "../stores/studio";
import { ODE_LANGUAGE_ID, registerODELanguage } from "../ode-language";

function handleEditorMount(monaco: Monaco) {
  registerODELanguage(monaco as unknown as typeof import("monaco-editor"));
}

export default function EquationEditor() {
  const { equations, threshold, reset, setEquations, setThreshold, setReset } =
    useStudioStore();

  const text = [
    ...equations,
    "",
    threshold ? `# threshold: ${threshold}` : "# threshold: (none)",
    reset ? `# reset: ${reset}` : "# reset: (none)",
  ].join("\n");

  function handleChange(value: string | undefined) {
    if (!value) return;
    const lines = value.split("\n");
    const eqLines: string[] = [];
    let newThreshold = threshold;
    let newReset = reset;

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith("# threshold:")) {
        newThreshold = trimmed.replace("# threshold:", "").trim();
        if (newThreshold === "(none)") newThreshold = "";
      } else if (trimmed.startsWith("# reset:")) {
        newReset = trimmed.replace("# reset:", "").trim();
        if (newReset === "(none)") newReset = "";
      } else if (trimmed.startsWith("d") && trimmed.includes("/dt")) {
        eqLines.push(trimmed);
      }
    }

    if (eqLines.length > 0) setEquations(eqLines);
    setThreshold(newThreshold);
    setReset(newReset);
  }

  return (
    <div className="monaco-wrapper">
      <Editor
        height="180px"
        defaultLanguage={ODE_LANGUAGE_ID}
        value={text}
        onChange={handleChange}
        beforeMount={handleEditorMount}
        options={{
          minimap: { enabled: false },
          lineNumbers: "off",
          fontSize: 13,
          fontFamily: "var(--font-mono)",
          scrollBeyondLastLine: false,
          wordWrap: "on",
          padding: { top: 8, bottom: 8 },
          renderLineHighlight: "none",
          overviewRulerLanes: 0,
          hideCursorInOverviewRuler: true,
          scrollbar: { vertical: "hidden", horizontal: "hidden" },
        }}
        theme="sc-ode-dark"
      />
    </div>
  );
}
