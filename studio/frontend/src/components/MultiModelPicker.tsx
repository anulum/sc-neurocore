// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { useId, useState } from "react";
import { Btn } from "../appChrome";
import { useStudioStore } from "../stores/studio";

/** How many models the overlay draws at once. */
export const MULTI_MODEL_LIMIT = 4;

/**
 * The models a filter keeps, chosen ones first so they never scroll away.
 *
 * The list used to be the first fifty models of the catalogue, so 135 of the
 * 185 could not be overlaid at all; every model is offered now and the filter
 * is what makes the list short.
 *
 * @param names - Every model the catalogue lists.
 * @param chosen - The models already picked.
 * @param filter - Text the name must contain, ignoring case.
 * @returns The names to offer, in order.
 */
export function multiModelChoices(names: readonly string[], chosen: readonly string[], filter: string): string[] {
  const needle = filter.trim().toLowerCase();
  const matches = names.filter((name) => !chosen.includes(name) && name.toLowerCase().includes(needle));
  return [...chosen.filter((name) => names.includes(name)), ...matches];
}

/**
 * Choose which models the overlay view runs together.
 *
 * @returns The panel.
 */
export default function MultiModelPicker() {
  const { models, selectedModelName, runMultiSimulate, isSimulating } = useStudioStore();
  const [selected, setSelected] = useState<string[]>([]);
  const [filter, setFilter] = useState("");
  const filterId = useId();
  const choices = multiModelChoices(models.map((m) => m.name), selected, filter);

  /**
   * Add or remove one model from the overlay.
   *
   * @param name - The model to toggle.
   */
  function toggle(name: string) {
    setSelected((prev) =>
      prev.includes(name)
        ? prev.filter((n) => n !== name)
        : prev.length < MULTI_MODEL_LIMIT
          ? [...prev, name]
          : prev
    );
  }

  /** Run the chosen models, or just the selected one when none are chosen. */
  function run() {
    const names = selected.length > 0 ? selected : [selectedModelName];
    void runMultiSimulate(names);
  }

  const full = selected.length >= MULTI_MODEL_LIMIT;
  return (
    <div className="panel-section">
      <h2 className="panel-header">Multi-model ({selected.length} of {MULTI_MODEL_LIMIT})</h2>
      <label htmlFor={filterId} className="visually-hidden">Filter models to overlay</label>
      <input
        id={filterId}
        type="search"
        value={filter}
        placeholder={`Filter ${models.length} models`}
        onChange={(e) => { setFilter(e.target.value); }}
        style={{ width: "100%", marginBottom: 4 }}
      />
      <div role="group" aria-label="Models to overlay" style={{ maxHeight: 120, overflowY: "auto", marginBottom: 6 }}>
        {choices.map((name) => {
          const checked = selected.includes(name);
          return (
            <label key={name} style={{
              display: "flex", alignItems: "center", gap: 6,
              fontSize: "var(--fs-body)", fontFamily: "var(--font-mono)",
              padding: "1px 4px", cursor: !checked && full ? "not-allowed" : "pointer",
              color: checked ? "var(--accent)" : "var(--text-muted)",
            }}>
              <input type="checkbox" checked={checked} disabled={!checked && full}
                onChange={() => { toggle(name); }} />
              {name}
            </label>
          );
        })}
        {choices.length === 0 && (
          <p className="panel-note" style={{ margin: "2px 4px" }}>No model name contains “{filter}”.</p>
        )}
      </div>
      <Btn
        label={selected.length > 0 ? `Overlay ${selected.length}` : "Overlay current"}
        onClick={run}
        disabled={isSimulating}
        outline
        title="Run the chosen models on the current drive and draw them in the Multi-model view"
      />
    </div>
  );
}
