import Editor from "@monaco-editor/react";
import { useStudioStore } from "../stores/studio";

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
    <Editor
      height="200px"
      defaultLanguage="plaintext"
      value={text}
      onChange={handleChange}
      options={{
        minimap: { enabled: false },
        lineNumbers: "off",
        fontSize: 14,
        scrollBeyondLastLine: false,
        wordWrap: "on",
      }}
      theme="vs-dark"
    />
  );
}
