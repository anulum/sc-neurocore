// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Trace exports and spike-triggered averages from the current run

import type { StudioState } from "./studioTypes";
import type { SimulationExportKind } from "../simulationExports";
import { simulationExportPlan } from "../simulationExports";
import { fullTraceUnavailableReason } from "../simulationRaw";
import { studioAnalysisErrorState, studioSTAResultState } from "../studioAnalysisState";

/**
 * Export the current run, reporting an explicitly unavailable raw trace.
 *
 * @param kind - Requested file format.
 * @param get - Read the current public store state.
 * @param set - Apply a result or user-visible explanation.
 */
export function exportStudioTrace(
  kind: SimulationExportKind,
  get: () => StudioState,
  set: (patch: Partial<StudioState>) => void,
): void {
  const plan = simulationExportPlan(kind, get().result);
  if (plan.available) {
    plan.writeArtefact();
  } else {
    if (plan.reason !== undefined) set(studioAnalysisErrorState(plan.reason));
    plan.runFallback();
  }
}

/**
 * Average full-resolution state samples on the run's original spike clock.
 *
 * @param get - Read the current public store state.
 * @param set - Apply the average or a user-visible explanation.
 */
export function computeStudioSTA(
  get: () => StudioState,
  set: (patch: Partial<StudioState>) => void,
): void {
  const { result } = get();
  if (result === null) return;
  const reason = fullTraceUnavailableReason(result);
  if (reason !== null) {
    set(studioAnalysisErrorState(`Spike-triggered average requires full-resolution traces. ${reason}`));
    return;
  }
  const state = studioSTAResultState(result);
  if (state !== null) set(state);
}
