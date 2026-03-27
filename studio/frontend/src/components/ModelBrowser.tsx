import { useEffect, useMemo, useState } from "react";
import { useStudioStore } from "../stores/studio";
import { fetchModelScan, type ModelBehavior } from "../api/client";

const PATTERN_COLORS: Record<string, string> = {
  tonic: "#81c784",
  bursting: "#ffb74d",
  adapting: "#4fc3f7",
  irregular: "#ce93d8",
  chaotic: "#ff5252",
  silent: "#484f58",
  single_spike: "#90a4ae",
  error: "#616161",
};

export default function ModelBrowser() {
  const {
    models, selectedModelName, modelFilter,
    loadModels, selectModel, setModelFilter,
  } = useStudioStore();

  const [behaviors, setBehaviors] = useState<Record<string, ModelBehavior>>({});
  const [patternFilter, setPatternFilter] = useState<string>("");
  const [scanLoaded, setScanLoaded] = useState(false);

  useEffect(() => { loadModels(); }, [loadModels]);

  function loadBehaviors() {
    if (scanLoaded) return;
    setScanLoaded(true);
    fetchModelScan().then((data) => {
      const map: Record<string, ModelBehavior> = {};
      for (const b of data) map[b.name] = b;
      setBehaviors(map);
    }).catch(() => setScanLoaded(false));
  }

  const grouped = useMemo(() => {
    let filtered = models;
    if (modelFilter) {
      const q = modelFilter.toLowerCase();
      filtered = filtered.filter((m) =>
        m.name.toLowerCase().includes(q) || m.category.toLowerCase().includes(q));
    }
    if (patternFilter) {
      filtered = filtered.filter((m) =>
        behaviors[m.name]?.pattern === patternFilter);
    }
    const groups: Record<string, typeof filtered> = {};
    for (const m of filtered) {
      (groups[m.category] ??= []).push(m);
    }
    return groups;
  }, [models, modelFilter, patternFilter, behaviors]);

  const totalFiltered = Object.values(grouped).reduce((s, g) => s + g.length, 0);
  const patterns = [...new Set(Object.values(behaviors).map((b) => b.pattern))].sort();

  return (
    <div>
      <div style={{ display: "flex", gap: 4, marginBottom: 4 }}>
        <input
          type="text"
          placeholder="Search models..."
          value={modelFilter}
          onChange={(e) => setModelFilter(e.target.value)}
          style={{
            flex: 1, padding: "4px 6px", fontSize: 11,
            background: "var(--bg-tertiary)", color: "var(--text-primary)",
            border: "1px solid var(--border)", borderRadius: "var(--radius)",
            outline: "none", fontFamily: "var(--font-mono)",
          }}
        />
        <button onClick={loadBehaviors} style={{
          fontSize: 9, padding: "2px 6px", background: "var(--bg-tertiary)",
          color: scanLoaded ? "var(--accent)" : "var(--text-muted)",
          border: "1px solid var(--border)", borderRadius: 3, cursor: "pointer",
        }} title="Classify all models by firing pattern (takes ~30s first time)">
          {scanLoaded ? "Scanned" : "Scan"}
        </button>
      </div>

      {patterns.length > 0 && (
        <div style={{ display: "flex", gap: 3, flexWrap: "wrap", marginBottom: 4 }}>
          <span onClick={() => setPatternFilter("")} style={{
            fontSize: 9, padding: "1px 5px", borderRadius: 3, cursor: "pointer",
            background: !patternFilter ? "var(--accent)" : "var(--bg-tertiary)",
            color: !patternFilter ? "var(--bg-primary)" : "var(--text-muted)",
          }}>all</span>
          {patterns.map((p) => (
            <span key={p} onClick={() => setPatternFilter(p === patternFilter ? "" : p)} style={{
              fontSize: 9, padding: "1px 5px", borderRadius: 3, cursor: "pointer",
              background: p === patternFilter ? (PATTERN_COLORS[p] || "var(--accent)") : "var(--bg-tertiary)",
              color: p === patternFilter ? "var(--bg-primary)" : (PATTERN_COLORS[p] || "var(--text-muted)"),
            }}>{p}</span>
          ))}
        </div>
      )}

      <div style={{ maxHeight: 200, overflowY: "auto" }}>
        {Object.entries(grouped).sort(([a], [b]) => a.localeCompare(b)).map(([cat, ms]) => (
          <div key={cat}>
            <div style={{
              fontSize: 9, fontWeight: 700, color: "var(--accent)",
              padding: "3px 4px 1px", textTransform: "uppercase", letterSpacing: "0.05em",
            }}>{cat} ({ms.length})</div>
            {ms.map((m) => {
              const beh = behaviors[m.name];
              return (
                <div key={m.name} onClick={() => selectModel(m.name)}
                  title={m.description || m.name}
                  style={{
                  padding: "2px 8px", fontSize: 10, fontFamily: "var(--font-mono)",
                  cursor: "pointer", borderRadius: 3,
                  background: m.name === selectedModelName ? "var(--accent-dim)" : "transparent",
                  color: m.name === selectedModelName ? "var(--accent)" : "var(--text-secondary)",
                  display: "flex", justifyContent: "space-between", alignItems: "center",
                }}>
                  <span>{m.name.replace("Neuron", "").replace("Model", "")}</span>
                  <span style={{ display: "flex", gap: 4, alignItems: "center" }}>
                    {beh && (
                      <span style={{
                        fontSize: 8, padding: "0 3px", borderRadius: 2,
                        background: PATTERN_COLORS[beh.pattern] || "var(--bg-tertiary)",
                        color: "var(--bg-primary)", fontWeight: 600,
                      }}>{beh.pattern}</span>
                    )}
                    <span style={{ color: "var(--text-muted)", fontSize: 9 }}>
                      {m.state_var_names.join(",")}&middot;{m.n_params}p
                    </span>
                  </span>
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <div style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 3 }}>
        {totalFiltered}/{models.length} models
        {Object.keys(behaviors).length > 0 && ` · ${Object.values(behaviors).filter((b) => b.pattern === "tonic").length} tonic · ${Object.values(behaviors).filter((b) => b.pattern === "bursting").length} bursting · ${Object.values(behaviors).filter((b) => b.pattern === "silent").length} silent`}
      </div>
    </div>
  );
}
