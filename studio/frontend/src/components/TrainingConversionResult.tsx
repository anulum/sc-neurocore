// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Conversion run result display

import type { TrainingConversionResult as Result } from "../trainingConversion";

/**
 * Format a fraction as a percentage with one decimal.
 *
 * @param value - A fraction.
 * @returns The percentage text.
 */
function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/**
 * State how the converted network did on the validation split, beside its source.
 *
 * @param props - The result the server reported, or `null` for a spiking run, and the run's target.
 * @returns The result line, or nothing.
 */
export default function TrainingConversionResult({ result, target }: { result: Result | null; target?: string }) {
  if (result === null) return null;
  return <p role="status" aria-label="Conversion result" style={{ padding: "0 12px", fontSize: "var(--fs-body)" }}>
    <strong>Converted network: {percent(result.val_accuracy)} validation accuracy</strong>
    {` (source ANN ${percent(result.source_val_accuracy)}, drop ${percent(result.conversion_accuracy_drop)}). `}
    Measured on the validation split and sealed in training/conversion_report.json.
    {result.target_accuracy !== undefined && ` With coefficients rounded for ${target ?? "the target"}: `
      + `${percent(result.target_accuracy)}, sealed in training/target_report.json.`}
  </p>;
}
