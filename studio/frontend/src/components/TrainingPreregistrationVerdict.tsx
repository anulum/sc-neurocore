// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Preregistered training verdict display

import type { TrainingPreregistrationVerdict as Verdict } from "../api/client";

const METRIC_LABELS = {
  val_accuracy: "validation accuracy",
  val_loss: "validation loss",
  conversion_accuracy_drop: "conversion accuracy drop",
} as const;

/**
 * State whether a finished run met the criterion stored before it started.
 *
 * @param props - The verdict the server reported, or `null` when none was declared or readable.
 * @returns The verdict line, or nothing.
 */
export default function TrainingPreregistrationVerdict({ verdict }: { verdict: Verdict | null }) {
  if (verdict === null) return null;
  const comparison = verdict.direction === "at_least" ? "≥" : "≤";
  const observed = verdict.observed === null ? "not a finite number" : String(verdict.observed);
  return <p role="status" aria-label="Preregistered verdict" style={{ padding: "0 12px", fontSize: 11 }}>
    <strong>{verdict.passed ? "Criterion met" : "Criterion missed"}</strong>
    {`: ${METRIC_LABELS[verdict.metric]} ${comparison} ${String(verdict.threshold)}, observed ${observed}. `}
    <span title={verdict.preregistration_sha256}>
      Criterion stored before the run as {verdict.preregistration_sha256.slice(0, 12)}.
    </span>
  </p>;
}
