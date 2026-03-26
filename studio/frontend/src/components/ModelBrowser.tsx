import { useEffect } from "react";
import { useStudioStore } from "../stores/studio";

export default function ModelBrowser() {
  const {
    models, selectedModelName, modelFilter,
    loadModels, selectModel, setModelFilter,
  } = useStudioStore();

  useEffect(() => { loadModels(); }, [loadModels]);

  const filtered = modelFilter
    ? models.filter((m) =>
        m.name.toLowerCase().includes(modelFilter.toLowerCase()) ||
        m.module.toLowerCase().includes(modelFilter.toLowerCase()))
    : models;

  return (
    <div>
      <input
        type="text"
        placeholder="Search 118 models..."
        value={modelFilter}
        onChange={(e) => setModelFilter(e.target.value)}
        style={{
          width: "100%",
          padding: "6px 8px",
          fontSize: 12,
          background: "var(--bg-tertiary)",
          color: "var(--text-primary)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          marginBottom: 6,
          outline: "none",
          fontFamily: "var(--font-mono)",
        }}
      />
      <div style={{ maxHeight: 200, overflowY: "auto" }}>
        {filtered.map((m) => (
          <div
            key={m.name}
            onClick={() => selectModel(m.name)}
            style={{
              padding: "4px 6px",
              fontSize: 11,
              fontFamily: "var(--font-mono)",
              cursor: "pointer",
              borderRadius: 3,
              background: m.name === selectedModelName ? "var(--accent-dim)" : "transparent",
              color: m.name === selectedModelName ? "var(--accent)" : "var(--text-secondary)",
              display: "flex",
              justifyContent: "space-between",
            }}
          >
            <span>{m.name}</span>
            <span style={{ color: "var(--text-muted)", fontSize: 10 }}>
              {m.state_var_names.join(",")} &middot; {m.n_params}p
            </span>
          </div>
        ))}
      </div>
      <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 4 }}>
        {filtered.length}/{models.length} models
      </div>
    </div>
  );
}
