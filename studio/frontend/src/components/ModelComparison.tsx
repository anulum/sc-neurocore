import { useMemo, useState } from "react";
import { useStudioStore } from "../stores/studio";
import { buildComparisonRows } from "../modelComparison";

const SHORT = (name: string) => name.replace(/Neuron$|Model$/, "");

export default function ModelComparison() {
  const { models, selectedModelName } = useStudioStore();
  const [picked, setPicked] = useState<string[]>([]);

  const selection = picked.length > 0 ? picked : selectedModelName ? [selectedModelName] : [];
  const chosen = useMemo(
    () => selection.map((n) => models.find((m) => m.name === n)).filter((m) => m !== undefined),
    [selection, models],
  );
  const rows = useMemo(() => buildComparisonRows(chosen), [chosen]);

  function toggle(name: string) {
    setPicked((prev) =>
      prev.includes(name)
        ? prev.filter((n) => n !== name)
        : prev.length < 4
          ? [...prev, name]
          : prev,
    );
  }

  return (
    <div className="panel-section">
      <div className="panel-header">Side-by-side ({chosen.length})</div>
      <div style={{ maxHeight: 110, overflowY: "auto", marginBottom: 4 }}>
        {models.map((m) => (
          <label key={m.name} style={{
            display: "flex", alignItems: "center", gap: 6, fontSize: 10,
            fontFamily: "var(--font-mono)", padding: "0 4px", cursor: "pointer",
            color: selection.includes(m.name) ? "var(--accent)" : "var(--text-muted)",
          }}>
            <input type="checkbox" checked={picked.includes(m.name)}
              onChange={() => toggle(m.name)} style={{ width: 11, height: 11 }} />
            {SHORT(m.name)}
          </label>
        ))}
      </div>
      {chosen.length > 0 && (
        <div style={{ overflowX: "auto" }}>
          <table style={{ borderCollapse: "collapse", fontSize: 9, width: "100%" }}>
            <thead>
              <tr>
                <th style={{ textAlign: "left", color: "var(--text-muted)", padding: "1px 4px" }} />
                {chosen.map((m) => (
                  <th key={m.name} style={{
                    textAlign: "left", padding: "1px 4px", color: "var(--accent)",
                    fontFamily: "var(--font-mono)", whiteSpace: "nowrap",
                  }}>{SHORT(m.name)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.label}>
                  <td style={{ color: "var(--text-muted)", padding: "1px 4px", whiteSpace: "nowrap" }}>
                    {row.label}
                  </td>
                  {row.values.map((v, i) => (
                    <td key={i} style={{
                      padding: "1px 4px", color: "var(--text-secondary)",
                      fontFamily: "var(--font-mono)", whiteSpace: "nowrap",
                    }}>{v}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
