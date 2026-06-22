import { useEffect, useState } from "react";
import { fetchModelDoc, type ModelDoc } from "../api/client";
import { useStudioStore } from "../stores/studio";

type DocState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; doc: ModelDoc }
  | { status: "absent" };

export default function ModelDocViewer() {
  const { sourceMode, selectedModelName } = useStudioStore();
  const [open, setOpen] = useState(false);
  const [state, setState] = useState<DocState>({ status: "idle" });

  useEffect(() => {
    if (!open || sourceMode !== "model" || !selectedModelName) return;
    let cancelled = false;
    setState({ status: "loading" });
    fetchModelDoc(selectedModelName)
      .then((doc) => {
        if (!cancelled) setState({ status: "ready", doc });
      })
      .catch(() => {
        if (!cancelled) setState({ status: "absent" });
      });
    return () => {
      cancelled = true;
    };
  }, [open, sourceMode, selectedModelName]);

  // Collapse and reset whenever the selected model changes.
  useEffect(() => {
    setOpen(false);
    setState({ status: "idle" });
  }, [selectedModelName]);

  if (sourceMode !== "model" || !selectedModelName) return null;

  return (
    <div className="panel-section">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        style={{
          width: "100%", textAlign: "left", fontSize: 10, padding: "3px 4px",
          background: "var(--bg-tertiary)", color: "var(--text-secondary)",
          border: "1px solid var(--border)", borderRadius: "var(--radius)", cursor: "pointer",
        }}
      >
        {open ? "▾" : "▸"} Reference documentation
      </button>
      {open && (
        <div style={{ marginTop: 4 }}>
          {state.status === "loading" && (
            <div style={{ fontSize: 10, color: "var(--text-muted)" }}>Loading…</div>
          )}
          {state.status === "absent" && (
            <div style={{ fontSize: 10, color: "var(--text-muted)" }}>
              No reference page for this model yet.
            </div>
          )}
          {state.status === "ready" && (
            <pre
              style={{
                maxHeight: 320, overflowY: "auto", margin: 0, padding: 8,
                fontSize: 10, lineHeight: 1.5, whiteSpace: "pre-wrap", wordBreak: "break-word",
                background: "var(--bg-primary)", color: "var(--text-secondary)",
                border: "1px solid var(--border)", borderRadius: "var(--radius)",
                fontFamily: "var(--font-mono)",
              }}
            >
              {state.doc.markdown}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
