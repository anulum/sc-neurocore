// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { useStudioStore } from "../stores/studio";

export default function SpikeStats() {
  const { result, setActiveTab } = useStudioStore();
  if (!result) return null;
  const { stats, spike_count } = result;

  return (
    <div className="model-info" style={{ flexDirection: "column", gap: 4 }}>
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        <div className="info-item">
          <span className="info-label">spikes:</span>
          <span className="info-value">{spike_count}</span>
        </div>
        <div className="info-item">
          <span className="info-label">rate:</span>
          <span className="info-value">{stats.rate_hz} Hz</span>
        </div>
        {stats.isi_mean_ms !== null && (
          <div className="info-item">
            <span className="info-label">ISI:</span>
            <span className="info-value">{stats.isi_mean_ms} ms</span>
          </div>
        )}
        {stats.isi_cv !== null && (
          <div className="info-item">
            <span className="info-label">CV:</span>
            <span className="info-value" style={{
              color: stats.isi_cv < 0.3 ? "var(--success)" :
                     stats.isi_cv < 0.7 ? "var(--warning)" : "var(--error)",
            }}>
              {stats.isi_cv} {stats.isi_cv < 0.3 ? "(regular)" :
                              stats.isi_cv < 0.7 ? "(irregular)" : "(bursting)"}
            </span>
          </div>
        )}
      </div>
      {stats.isi_histogram && (
        <div style={{ fontSize: 10, color: "var(--accent)", cursor: "pointer" }}
          onClick={() => setActiveTab("isi")}>
          View ISI histogram →
        </div>
      )}
    </div>
  );
}
