// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Preregistered training criterion form

import type { TrainingModelKind, TrainingPreregistration, TrainingPreregistrationMetric } from "../api/client";
import { PREREGISTRATION_RATIONALE_MAX, preregistrationProblem } from "../trainingPreregistration";

/** The criterion being edited, and where edits go. */
export interface TrainingPreregistrationInputProps {
  value: TrainingPreregistration | undefined;
  onChange: (value: TrainingPreregistration | undefined) => void;
  /** The run's kind; only a conversion run offers the accuracy-drop criterion. */
  modelKind?: TrainingModelKind;
}

const DEFAULT_CRITERION: TrainingPreregistration = { metric: "val_accuracy", threshold: 0.5, rationale: "" };

/**
 * Declare how a run will be judged before it starts, or declare nothing.
 *
 * @param props - The current criterion and the change handler.
 * @returns The criterion controls and, when the criterion cannot be sent, why.
 */
export default function TrainingPreregistrationInput({ value, onChange, modelKind }: TrainingPreregistrationInputProps) {
  const problem = value === undefined ? null : preregistrationProblem(value, modelKind);
  return <fieldset style={{ margin: "0 12px 8px", fontSize: 10, border: "1px solid var(--border)" }}>
    <legend style={{ color: "var(--text-secondary)" }}>Preregistered acceptance criterion</legend>
    <label style={{ display: "flex", alignItems: "center", gap: 4 }}>
      <input type="checkbox" checked={value !== undefined}
        onChange={(e) => { onChange(e.target.checked ? { ...DEFAULT_CRITERION } : undefined); }} />
      Judge this run against a criterion stored before it starts
    </label>
    {value !== undefined && <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginTop: 6 }}>
      <label>
        Metric
        <select aria-label="Criterion metric" value={value.metric}
          onChange={(e) => { onChange({ ...value, metric: e.target.value as TrainingPreregistrationMetric }); }}
          style={{ display: "block", width: "100%", fontSize: 10 }}>
          <option value="val_accuracy">Validation accuracy, at least</option>
          <option value="val_loss">Validation loss, at most</option>
          {modelKind === "qcfs_conversion"
            && <option value="conversion_accuracy_drop">Conversion accuracy drop, at most</option>}
        </select>
      </label>
      <label>
        Threshold
        <input aria-label="Criterion threshold" type="number" step="any" value={value.threshold}
          onChange={(e) => { onChange({ ...value, threshold: e.target.value === "" ? Number.NaN : Number(e.target.value) }); }}
          style={{ display: "block", width: "100%", fontSize: 10 }} />
      </label>
      <label style={{ gridColumn: "1 / -1" }}>
        Rationale
        <textarea aria-label="Criterion rationale" value={value.rationale} maxLength={PREREGISTRATION_RATIONALE_MAX} rows={2}
          onChange={(e) => { onChange({ ...value, rationale: e.target.value }); }}
          style={{ display: "block", width: "100%", fontSize: 10 }} />
      </label>
    </div>}
    {problem !== null && <p role="alert" style={{ margin: "4px 0 0" }}>{problem}</p>}
  </fieldset>;
}
