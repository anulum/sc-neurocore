// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — What a co-simulation result establishes, in words

import type { ModelCosimReport } from "./api/client";

/** The verdict line, the run it covers, a mismatch if any, and its caveats. */
export interface CosimVerdict {
  pass: boolean;
  label: string;
  cycles: string;
  mismatch: string | null;
  title: string;
}

/**
 * Say what a co-simulation compared and what its agreement does not show.
 *
 * Agreement means the RTL and the generated bit-true C kernel produced equal
 * traces for the stimuli run. It is neither the model's scientific fidelity
 * nor a proof for other stimuli, and the verdict says so.
 *
 * @param report - The co-simulation report.
 * @returns The verdict to render.
 */
export function cosimVerdict(report: ModelCosimReport): CosimVerdict {
  const stress = report.stress;
  const mismatch = report.first_mismatch
    ? `cycle ${String(report.first_mismatch.cycle)}: ${report.first_mismatch.signals.join(", ")}`
    : stress?.first_mismatch
      ? `stress cycle ${String(stress.first_mismatch.cycle)}: ${stress.first_mismatch.signals.join(", ")}`
      : null;
  const caveats = [
    report.boundary?.statement
      ?? "the RTL against the generated bit-true C kernel for the stimulus run",
    ...(report.boundary?.not_covered ?? []).map((item) => `Not covered: ${item}.`),
    "Not the model's scientific fidelity, and not a proof.",
  ];
  return {
    pass: report.bit_exact,
    label: report.bit_exact ? "RTL = bit-true C kernel" : "RTL ≠ bit-true C kernel",
    cycles: stress
      ? `${String(report.sample_count)} requested + ${String(stress.sample_count)} stress cycles`
      : `${String(report.sample_count)} cycles`,
    mismatch,
    title: caveats.join(" "),
  };
}
