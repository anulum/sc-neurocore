// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio Zustand store
// Async heavy-analysis orchestration for the Studio store (W12-D path).

import type { AnalysisJobKind } from "../api/client";
import { buildStudioAnalysisJobSelection } from "../studioAnalysisJobSelection";
import { runStudioAnalysisJob } from "../studioAnalysisJobRunner";
import { studioAnalysisFailureState } from "../studioAnalysisState";
import { studioExperimentKey } from "../studioExperimentKey";
import type { StudioSimulationConfigInput } from "../studioSimulationConfig";
import type { StudioState } from "./studioTypes";

/**
 * Take the simulation configuration out of the store.
 *
 * Named fields rather than a spread, so a field added to the store does not
 * silently become part of every analysis request.
 *
 * @param s - The store's current state.
 * @returns The configuration a run is built from.
 */
export function simulationConfigInput(s: StudioState): StudioSimulationConfigInput {
  return {
    sourceMode: s.sourceMode,
    selectedModelName: s.selectedModelName,
    modelParams: s.modelParams,
    equations: s.equations,
    threshold: s.threshold,
    reset: s.reset,
    odeParams: s.odeParams,
    odeInit: s.odeInit,
    dt: s.dt,
    duration: s.duration,
    current: s.current,
    protocol: s.protocol,
    frequencyHz: s.frequencyHz,
    seed: s.seed,
    trial: s.trial,
  };
}

/**
 * Derive an experiment identity or report invalid input without submitting.
 *
 * @param state - The current experiment configuration.
 * @param set - Reports invalid configuration and withdraws completion evidence.
 * @returns The key, or null when the configuration cannot be identified.
 */
function experimentKeyOrFailure(
  state: StudioState,
  set: (partial: Partial<StudioState>) => void,
): string | null {
  try {
    return studioExperimentKey(simulationConfigInput(state));
  } catch (error: unknown) {
    set({ ...studioAnalysisFailureState(error), analysisExperimentKey: null });
    return null;
  }
}

/**
 * Run one heavy analysis as a job, and write its result into the store.
 *
 * These are the analyses that take long enough to need a job rather than a
 * request. The guards at the top are refusals, not validation: a sweep with no
 * parameter chosen cannot be built, and starting one while a run is already in
 * flight would leave two sets of results racing for the same fields.
 *
 * A request that fails to build is written to the store as a failure. A job
 * that ran and failed is not written again here -- the runner has already
 * applied its own failure patch, and writing a second would replace the
 * runner's specific message with this one's.
 *
 * @param kind - Which analysis to run.
 * @param get - Reads the store.
 * @param set - Writes to the store.
 */
export async function runStoreHeavyAnalysis(
  kind: AnalysisJobKind,
  get: () => StudioState,
  set: (partial: Partial<StudioState>) => void,
): Promise<void> {
  const s = get();
  if (s.isSimulating) return;
  if (kind === "bifurcation" && !s.sweepParam) return;
  if (kind === "heatmap" && (!s.sweepParam || !s.sweepParamY)) return;
  if (kind === "heatmap") set({ heatmapExperimentKey: null });
  const selection = buildStudioAnalysisJobSelection({
    analysis: kind,
    sourceMode: s.sourceMode,
    modelParams: s.modelParams,
    odeParams: s.odeParams,
    sweepParam: s.sweepParam,
    sweepParamY: s.sweepParamY,
  });
  if (!selection.ok) {
    set({ ...studioAnalysisFailureState(selection.error), analysisExperimentKey: null });
    return;
  }
  // Only a successful result patch may attest this experiment. Start/failure
  // patches must not relabel a historical result as a newly completed analysis.
  const requestedKey = experimentKeyOrFailure(s, set);
  if (requestedKey === null) return;
  const outcome = await runStudioAnalysisJob(
    { simulation: simulationConfigInput(s), selection: selection.selection },
    {
      applyPatch: (patch) => {
        if (experimentKeyOrFailure(get(), set) !== requestedKey) {
          if (!patch.isSimulating) set({ isSimulating: false });
          return;
        }
        // The result sink clears error and ends the run only after validation;
        // start patches are busy, and failure patches carry a non-null error.
        const completed = !patch.isSimulating && "error" in patch && patch.error === null;
        set({ ...patch, analysisExperimentKey: completed ? requestedKey : null,
          ...(kind === "heatmap" ? { heatmapExperimentKey: completed ? requestedKey : null } : {}) });
      },
    },
  );
  if (!outcome.ok && outcome.stage === "request") {
    set({ ...studioAnalysisFailureState(outcome.error), analysisExperimentKey: null });
  }
}
