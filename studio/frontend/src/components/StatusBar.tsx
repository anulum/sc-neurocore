// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { useStudioStore } from "../stores/studio";

export default function StatusBar() {
  const { result, activeTab, isSimulating, sourceMode, selectedModelName, networkResult } = useStudioStore();

  const parts: string[] = [];

  if (isSimulating) {
    parts.push("simulating...");
  } else if (activeTab === "network" && networkResult) {
    parts.push(`E-I Network: ${networkResult.n_exc}E/${networkResult.n_inh}I`);
    parts.push(`${networkResult.n_spikes} spikes`);
    parts.push(`E: ${networkResult.mean_exc_rate}Hz`);
    parts.push(`I: ${networkResult.mean_inh_rate}Hz`);
  } else if (result) {
    if (sourceMode === "model") parts.push(selectedModelName);
    parts.push(`${result.spike_count} spikes`);
    parts.push(`${result.stats.rate_hz} Hz`);
    if (result.pattern) parts.push(result.pattern.pattern);
    parts.push(`dt=${result.dt} T=${result.time[result.time.length - 1]}ms`);
  }

  if (parts.length === 0) return null;

  return (
    <div style={{
      height: 18, padding: "0 12px",
      display: "flex", alignItems: "center", gap: 12,
      background: "var(--bg-secondary)", borderTop: "1px solid var(--border)",
      fontSize: 9, fontFamily: "var(--font-mono)", color: "var(--text-muted)",
    }}>
      {parts.map((p, i) => (
        <span key={i}>{p}</span>
      ))}
      <span style={{ marginLeft: "auto" }}>? for shortcuts</span>
    </div>
  );
}
