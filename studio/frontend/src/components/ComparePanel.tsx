// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { useState } from "react";
import { useStudioStore } from "../stores/studio";

export default function ComparePanel() {
  const { activeTab, models, selectedModelName, duration, current, protocol, runCompare, isSimulating } = useStudioStore();
  const [compareModel, setCompareModel] = useState("");

  if (activeTab !== "trace" && activeTab !== "compare") return null;

  const available = models.filter((m) => m.name !== selectedModelName);

  function handleCompare() {
    if (!compareModel) return;
    runCompare({
      model_name: compareModel,
      params: {},
      duration,
      current,
      protocol,
    });
  }

  return (
    <div className="panel-section">
      <div className="panel-header">Compare</div>
      <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
        <select
          value={compareModel}
          onChange={(e) => setCompareModel(e.target.value)}
          style={{ flex: 1, fontSize: 10, padding: "2px 4px" }}
        >
          <option value="">Select model B...</option>
          {available.map((m) => (
            <option key={m.name} value={m.name}>{m.name}</option>
          ))}
        </select>
        <button
          className="btn-simulate"
          onClick={handleCompare}
          disabled={!compareModel || isSimulating}
          style={{
            background: "#ce93d8", color: "#0d1117", border: "none",
            padding: "2px 8px", fontSize: 10,
          }}
        >
          vs
        </button>
      </div>
    </div>
  );
}
