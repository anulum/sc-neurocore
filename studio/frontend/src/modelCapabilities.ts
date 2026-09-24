// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio model silicon capability rows

/**
 * Turning a model's capability matrix into the rows the model panel shows.
 *
 * An enabled operation states what it will run; a disabled one states why
 * not, in the server's words, so a reader never meets a greyed control with
 * no explanation.
 */

import type { ModelCapabilities, ModelCapabilityOperation } from "./api/client";

/** One row of the silicon capability strip. */
export interface CapabilityRow {
  label: string;
  enabled: boolean;
  detail: string;
}

const LABELS: [keyof ModelCapabilities["operations"], string][] = [
  ["compile", "compile"],
  ["cosimulate", "co-simulate"],
  ["synthesise", "synthesise"],
  ["place_and_route", "place & route"],
  ["formal", "formal"],
];

/**
 * Describe what one enabled operation will run.
 *
 * @param key - The operation.
 * @param operation - Its matrix entry.
 * @returns A short statement of what is offered.
 */
function enabledDetail(key: string, operation: ModelCapabilityOperation): string {
  if (key === "compile") {
    return `${(operation.integrators ?? []).join(", ")} · ${(operation.q_formats ?? []).join(", ")}`;
  }
  if (key === "cosimulate") {
    const mirrored = (operation.combinations ?? []).filter((combination) => combination.mirrored);
    return mirrored.map((combination) => `${combination.integrator} ${combination.q_format}`).join(", ");
  }
  return (operation.targets ?? []).join(", ");
}

/**
 * Turn a capability matrix into strip rows, in pipeline order.
 *
 * @param capabilities - The matrix the server returned.
 * @returns One row per operation.
 */
export function capabilityRows(capabilities: ModelCapabilities): CapabilityRow[] {
  return LABELS.map(([key, label]) => {
    const operation = capabilities.operations[key];
    if (operation.enabled) {
      return { label, enabled: true, detail: enabledDetail(key, operation) };
    }
    const job = operation.catalogue_job;
    const suffix = job ? ` (catalogue job ${job.module}: ${job.claim} to depth ${String(job.depth)})` : "";
    return { label, enabled: false, detail: `${operation.reason ?? "unavailable"}${suffix}` };
  });
}
