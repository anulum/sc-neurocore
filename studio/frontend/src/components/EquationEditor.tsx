// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import Editor from "@monaco-editor/react";
import { useRef } from "react";
import { useStudioStore } from "../stores/studio";
import { ODE_LANGUAGE_ID, registerODELanguage } from "../ode-language";

/**
 * Render the store's equations, threshold and reset as the editor's text.
 *
 * @param equations - The equation lines.
 * @param threshold - The threshold expression, empty when there is none.
 * @param reset - The reset expression, empty when there is none.
 * @returns The text: the equations, a blank line, then the two comment lines.
 */
function editorText(equations: readonly string[], threshold: string, reset: string): string {
  return [
    ...equations,
    "",
    threshold ? `# threshold: ${threshold}` : "# threshold: (none)",
    reset ? `# reset: ${reset}` : "# reset: (none)",
  ].join("\n");
}

/** What a text in the editor says, read the way the store takes it. */
interface ParsedEditorText {
  /** The equation lines, or null when the text holds none and the store keeps its own. */
  equations: string[] | null;
  threshold: string;
  reset: string;
}

/**
 * Read the editor's text into equations, threshold and reset.
 *
 * @param value - The editor's whole text.
 * @param threshold - The current threshold, kept when the text names none.
 * @param reset - The current reset, kept when the text names none.
 * @returns What the store should take from the text.
 */
function parseEditorText(value: string, threshold: string, reset: string): ParsedEditorText {
  const equations: string[] = [];
  let newThreshold = threshold;
  let newReset = reset;
  for (const line of value.split("\n")) {
    const trimmed = line.trim();
    if (trimmed.startsWith("# threshold:")) {
      newThreshold = trimmed.replace("# threshold:", "").trim();
      if (newThreshold === "(none)") newThreshold = "";
    } else if (trimmed.startsWith("# reset:")) {
      newReset = trimmed.replace("# reset:", "").trim();
      if (newReset === "(none)") newReset = "";
    } else if (trimmed.startsWith("d") && trimmed.includes("/dt")) {
      equations.push(trimmed);
    }
  }
  return { equations: equations.length > 0 ? equations : null, threshold: newThreshold, reset: newReset };
}

/** The last text a person typed, and the store's rendering that text leads to. */
interface TypedText {
  text: string;
  rendered: string;
}

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
 * The editor's text is the store's rendering, except while a person is
 * typing. The store keeps only what the text means (trimmed equation lines,
 * the threshold and the reset), so its rendering differs from the typed text
 * as soon as a line reads as an equation. Handing that rendering back would
 * replace the text under the caret and move the caret to the end, and the
 * rest of the typing would land in the reset line. So the typed text stays
 * for as long as the store holds exactly what it parses to. A change made
 * elsewhere, such as a chosen template or a loaded project, gives another
 * rendering: it is shown, and the typed text is over. A later return of the
 * store to what was typed, such as loading a project saved then, shows the
 * store's rendering, threshold and reset lines included.
 *
 * The typed text is a ref, written in the change handler before the store's
 * setters run. Every render after that keystroke reads it, however React
 * schedules the store's update and this component's own, so no render can
 * see the store's new text without it.
 *
 * @returns The editor.
 */
export default function EquationEditor() {
  const { equations, threshold, reset, setEquations, setThreshold, setReset } =
    useStudioStore();
  const typed = useRef<TypedText | null>(null);

  const rendered = editorText(equations, threshold, reset);
  // Another rendering ends the typed text for good. Dropping it is the same in
  // every render of this state, so a render React repeats or discards changes
  // nothing.
  if (typed.current !== null && typed.current.rendered !== rendered) typed.current = null;
  const text = typed.current === null ? rendered : typed.current.text;

  /**
   * Carry the edited text into the store, and keep it as the editor's text.
   *
   * @param value - The editor's contents, absent while it is loading.
   */
  function handleChange(value: string | undefined) {
    if (!value) return;
    const parsed = parseEditorText(value, threshold, reset);
    typed.current = {
      text: value,
      rendered: editorText(parsed.equations ?? equations, parsed.threshold, parsed.reset),
    };
    if (parsed.equations !== null) setEquations(parsed.equations);
    setThreshold(parsed.threshold);
    setReset(parsed.reset);
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
