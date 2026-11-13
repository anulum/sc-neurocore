// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { useStudioStore } from "../stores/studio";

/**
 * The bar along the bottom: what is running and what last happened.
 *
 * The bar is always drawn, and what it reports sits in one status region, so a
 * screen reader announces a run starting and finishing. A region that only
 * appears with its first result is not announced reliably. The shortcut hint
 * stays outside the region: it never changes, so it is not news. The parts are
 * set apart by the bar's gap on screen and by a comma read aloud, and the bar
 * wraps onto further lines in a narrow window.
 *
 * @returns The bar.
 */
export default function StatusBar() {
  const {
    result, activeTab, isSimulating, busyWith, sourceMode, selectedModelName, networkResult, graphSimResult,
  } = useStudioStore();

  const parts: string[] = [];

  if (isSimulating) {
    parts.push(`${busyWith.length > 0 ? busyWith : "Run"} running…`);
  } else if (activeTab === "canvas" && graphSimResult?.success) {
    // A count the result does not carry is shown as unknown, never as zero.
    parts.push(`Network: ${graphSimResult.n_total ?? "?"} neurons`);
    parts.push(`${graphSimResult.n_spikes ?? "?"} spikes`);
    parts.push(`${graphSimResult.n_steps ?? "?"} steps × ${graphSimResult.dt ?? "?"} ms`);
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

  return (
    <div style={{
      minHeight: 18, padding: "0 12px",
      display: "flex", flexWrap: "wrap", alignItems: "center", gap: 12,
      background: "var(--bg-secondary)", borderTop: "1px solid var(--border)",
      fontSize: "var(--fs-meta)", fontFamily: "var(--font-mono)", color: "var(--text-muted)",
    }}>
      <span role="status" style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
        {parts.map((p, i) => (
          <span key={i}>{i > 0 && <span className="visually-hidden">, </span>}{p}</span>
        ))}
      </span>
      <span style={{ marginLeft: "auto" }}>? for shortcuts</span>
    </div>
  );
}
