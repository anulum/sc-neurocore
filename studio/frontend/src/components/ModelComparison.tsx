// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { useMemo, useState } from "react";
import { useStudioStore } from "../stores/studio";
import { buildComparisonRows } from "../modelComparison";
import { shortModelLabels } from "../modelShortName";

/** How many models the side-by-side table holds. */
export const SIDE_BY_SIDE_LIMIT = 4;

/**
 * Choose which models the comparison view draws.
 *
 * The heading and the group are named for this panel: its checkboxes carry
 * the same model names as the multi-model picker's, and without a group name
 * a screen reader heard two identical lists. A fifth model is refused visibly
 * (the other boxes are disabled) rather than by ignoring the click.
 *
 * @returns The panel.
 */
export default function ModelComparison() {
  const { models, selectedModelName } = useStudioStore();
  const [picked, setPicked] = useState<string[]>([]);
  const labels = useMemo(() => shortModelLabels(models.map((m) => m.name)), [models]);
  const SHORT = (name: string) => labels.get(name) ?? name;

  const selection = picked.length > 0 ? picked : selectedModelName ? [selectedModelName] : [];
  const chosen = useMemo(
    () => selection.map((n) => models.find((m) => m.name === n)).filter((m) => m !== undefined),
    [selection, models],
  );
  const rows = useMemo(() => buildComparisonRows(chosen), [chosen]);

  /**
   * Add or remove one model from the comparison.
   *
   * @param name - The model to toggle.
   */
  function toggle(name: string) {
    setPicked((prev) =>
      prev.includes(name)
        ? prev.filter((n) => n !== name)
        : prev.length < SIDE_BY_SIDE_LIMIT
          ? [...prev, name]
          : prev,
    );
  }

  const full = picked.length >= SIDE_BY_SIDE_LIMIT;
  return (
    <div className="panel-section">
      <h2 className="panel-header">Side-by-side ({chosen.length} of {SIDE_BY_SIDE_LIMIT})</h2>
      <div role="group" aria-label="Models to compare side by side"
        style={{ maxHeight: 110, overflowY: "auto", marginBottom: 4 }}>
        {models.map((m) => {
          const checked = picked.includes(m.name);
          return (
            <label key={m.name} style={{
              display: "flex", alignItems: "center", gap: 6, fontSize: "var(--fs-body)",
              fontFamily: "var(--font-mono)", padding: "0 4px",
              cursor: !checked && full ? "not-allowed" : "pointer",
              color: selection.includes(m.name) ? "var(--accent)" : "var(--text-muted)",
            }}>
              <input type="checkbox" checked={checked} disabled={!checked && full}
                onChange={() => { toggle(m.name); }} style={{ width: 11, height: 11 }} />
              {SHORT(m.name)}
            </label>
          );
        })}
      </div>
      {chosen.length > 0 && (
        <div style={{ overflowX: "auto" }}>
          <table style={{ borderCollapse: "collapse", fontSize: "var(--fs-meta)", width: "100%" }}>
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
