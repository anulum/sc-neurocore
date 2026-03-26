import { useEffect, useMemo } from "react";
import { useStudioStore } from "../stores/studio";

export default function ModelBrowser() {
  const {
    models, selectedModelName, modelFilter,
    loadModels, selectModel, setModelFilter,
  } = useStudioStore();

  useEffect(() => { loadModels(); }, [loadModels]);

  const grouped = useMemo(() => {
    const filtered = modelFilter
      ? models.filter((m) =>
          m.name.toLowerCase().includes(modelFilter.toLowerCase()) ||
          m.category.toLowerCase().includes(modelFilter.toLowerCase()))
      : models;
    const groups: Record<string, typeof filtered> = {};
    for (const m of filtered) {
      (groups[m.category] ??= []).push(m);
    }
    return groups;
  }, [models, modelFilter]);

  const totalFiltered = Object.values(grouped).reduce((s, g) => s + g.length, 0);

  return (
    <div>
      <input
        type="text"
        placeholder="Search 118 models..."
        value={modelFilter}
        onChange={(e) => setModelFilter(e.target.value)}
        style={{
          width: "100%", padding: "6px 8px", fontSize: 12,
          background: "var(--bg-tertiary)", color: "var(--text-primary)",
          border: "1px solid var(--border)", borderRadius: "var(--radius)",
          marginBottom: 6, outline: "none", fontFamily: "var(--font-mono)",
        }}
      />
      <div style={{ maxHeight: 220, overflowY: "auto" }}>
        {Object.entries(grouped).sort(([a], [b]) => a.localeCompare(b)).map(([cat, ms]) => (
          <div key={cat}>
            <div style={{
              fontSize: 10, fontWeight: 700, color: "var(--accent)",
              padding: "4px 4px 2px", textTransform: "uppercase", letterSpacing: "0.05em",
            }}>{cat} ({ms.length})</div>
            {ms.map((m) => (
              <div key={m.name} onClick={() => selectModel(m.name)} style={{
                padding: "3px 8px", fontSize: 11, fontFamily: "var(--font-mono)",
                cursor: "pointer", borderRadius: 3,
                background: m.name === selectedModelName ? "var(--accent-dim)" : "transparent",
                color: m.name === selectedModelName ? "var(--accent)" : "var(--text-secondary)",
                display: "flex", justifyContent: "space-between",
              }}>
                <span>{m.name.replace("Neuron", "").replace("Model", "")}</span>
                <span style={{ color: "var(--text-muted)", fontSize: 10 }}>
                  {m.state_var_names.join(",")}&middot;{m.n_params}p
                </span>
              </div>
            ))}
          </div>
        ))}
      </div>
      <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 4 }}>
        {totalFiltered}/{models.length} models
      </div>
    </div>
  );
}
