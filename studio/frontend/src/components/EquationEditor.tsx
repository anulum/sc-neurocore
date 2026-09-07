// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import Editor from "@monaco-editor/react";
import { useStudioStore } from "../stores/studio";
import { ODE_LANGUAGE_ID, registerODELanguage } from "../ode-language";

/**
 * Register the Studio's ODE language once the editor is up.
 *
 * @param monaco - The editor module, as the wrapper hands it over.
 */
function handleEditorMount(monaco: typeof import("monaco-editor")) {
  // The wrapper's own `Monaco` type is `any`, so taking it here would make
  // everything downstream unchecked. Naming the editor's module type instead
  // is narrower than what the prop supplies, which is allowed, and it is what
  // `registerODELanguage` actually requires.
  registerODELanguage(monaco);
}

/**
 * The custom-ODE editor, with the Studio's own language registered.
 *
 * @returns The editor.
 */
export default function EquationEditor() {
  const { equations, threshold, reset, setEquations, setThreshold, setReset } =
    useStudioStore();

  const text = [
    ...equations,
    "",
    threshold ? `# threshold: ${threshold}` : "# threshold: (none)",
    reset ? `# reset: ${reset}` : "# reset: (none)",
  ].join("\n");

  /**
   * Carry the edited text into the store.
   *
   * @param value - The editor's contents, absent while it is loading.
   */
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
