// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { useState } from "react";
import { useStudioStore } from "../stores/studio";

export default function MultiModelPicker() {
  const { models, selectedModelName, runMultiSimulate, isSimulating } = useStudioStore();
  const [selected, setSelected] = useState<string[]>([]);

  function toggle(name: string) {
    setSelected((prev) =>
      prev.includes(name)
        ? prev.filter((n) => n !== name)
        : prev.length < 4
          ? [...prev, name]
          : prev
    );
  }

  function run() {
    const names = selected.length > 0 ? selected : [selectedModelName];
    runMultiSimulate(names);
  }

  return (
    <div className="panel-section">
      <div className="panel-header">Compare Models (pick up to 4)</div>
      <div style={{ maxHeight: 120, overflowY: "auto", marginBottom: 6 }}>
        {models.slice(0, 50).map((m) => (
          <label key={m.name} style={{
            display: "flex", alignItems: "center", gap: 6,
            fontSize: 10, fontFamily: "var(--font-mono)",
            padding: "1px 4px", cursor: "pointer",
            color: selected.includes(m.name) ? "var(--accent)" : "var(--text-muted)",
          }}>
            <input type="checkbox" checked={selected.includes(m.name)}
              onChange={() => toggle(m.name)}
              style={{ width: 12, height: 12 }} />
            {m.name.replace("Neuron", "").replace("Model", "")}
          </label>
        ))}
      </div>
      <button onClick={run} disabled={isSimulating} style={{
        fontSize: 10, padding: "3px 10px", background: "#80cbc4",
        color: "var(--bg-primary)", border: "none", borderRadius: 3,
        cursor: isSimulating ? "wait" : "pointer", fontWeight: 600,
      }}>
        Compare {selected.length > 0 ? `(${selected.length})` : "current"}
      </button>
    </div>
  );
}
