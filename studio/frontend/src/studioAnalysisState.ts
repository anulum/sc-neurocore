// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio analysis store state helpers

/**
 * The state patches every analysis transition produces.
 *
 * Each function returns exactly the fields one transition changes, and each
 * has a named patch type, so a transition cannot quietly set a field it has no
 * business setting. That is the point of the pattern: the store applies these
 * and does not decide what they contain, which is what makes the transitions
 * testable without a store at all.
 *
 * `isSimulating` is what most of them are about. A start sets it, a result or
 * a failure clears it, and an error alone does not touch it -- an error while
 * something else is still running must not report the run as finished.
 */

import { at } from "./arrayAt";
import type {
  BifurcationResponse,
  CompareResponse,
  FICurveResponse,
  FreqResponse,
  HeatmapResponse,
  ImportedTrace,
  NetworkResult,
  NullclineResponse,
  PrecisionResponse,
  SensitivityResponse,
  SimulateResponse,
} from "./api/client";
import { fullStateNames, fullStateTrace } from "./simulationRaw";

/** An analysis has begun: the error clears, the run is marked in flight, and the tab moves if the caller named one. */
export interface StudioAnalysisStartStatePatch {
  activeTab?: "bifurcation" | "compare" | "fi-curve" | "freq" | "heatmap" | "multi" |
    "network" | "precision" | "sensitivity";
  error: null;
  isSimulating: true;
}

/** An analysis failed: the message is shown and the run is no longer in flight. */
export interface StudioAnalysisFailureStatePatch {
  error: string;
  isSimulating: false;
}

/** A message to show without claiming anything about what is running. */
export interface StudioAnalysisErrorStatePatch {
  error: string;
}

/** Nothing is in flight, said without touching the error or any result. */
export interface StudioAnalysisIdleStatePatch {
  isSimulating: false;
}

/** A finished run, with the fields the trace view reads. */
export interface StudioSimulationResultStatePatch {
  isSimulating: false;
  result: SimulateResponse;
}

/** A finished f-I sweep. */
export interface StudioFICurveResultStatePatch {
  fiResult: FICurveResponse;
  isSimulating: false;
}

/** A finished attractor sweep. */
export interface StudioBifurcationResultStatePatch {
  bifResult: BifurcationResponse;
  isSimulating: false;
}

/** A finished elasticity analysis. */
export interface StudioSensitivityResultStatePatch {
  isSimulating: false;
  sensResult: SensitivityResponse;
}

/** A finished precision comparison. */
export interface StudioPrecisionResultStatePatch {
  isSimulating: false;
  precResult: PrecisionResponse;
}

/** A finished two-parameter sweep. */
export interface StudioHeatmapResultStatePatch {
  heatmapResult: HeatmapResponse;
  isSimulating: false;
}

/** The code tab, opened before the script arrives. */
export interface StudioCodegenStartStatePatch {
  activeTab: "code";
}

/** A generated script, its one-line form, its replay form and the experiment digest it checks. */
export interface StudioCodegenResultStatePatch {
  codeExperimentSha256: string;
  codeOneliner: string;
  codeReplayScript: string;
  codeScript: string;
}

/** A finished multi-model overlay. */
export interface StudioMultiResultsStatePatch {
  isSimulating: false;
  multiResults: SimulateResponse[];
}

/** A finished network run. */
export interface StudioNetworkResultStatePatch {
  isSimulating: false;
  networkResult: NetworkResult;
}

/** A recorded trace taken in, with the trace tab opened to show it. */
export interface StudioImportedTraceStatePatch {
  activeTab: "trace";
  importedTrace: ImportedTrace;
}

/** A finished two-run comparison. */
export interface StudioCompareResultStatePatch {
  compareResult: CompareResponse;
  isSimulating: false;
}

/** Finished nullclines, with the phase tab opened to show them. */
export interface StudioNullclineResultStatePatch {
  activeTab: "phase";
  isSimulating: false;
  nullclineResult: NullclineResponse;
}

/** A finished frequency sweep. */
export interface StudioFrequencyResultStatePatch {
  freqResult: FreqResponse;
  isSimulating: false;
}

/** A computed spike-triggered average. */
export interface StudioSTAResultStatePatch {
  activeTab: "sta";
  staResult: {
    average: number[];
    n_spikes: number;
    time_ms: number[];
  };
}

/**
 * Mark an analysis as begun.
 *
 * @param activeTab - The tab to move to, or nothing to stay where the user is.
 * @returns The patch.
 */
export function studioAnalysisStartState(
  activeTab?: StudioAnalysisStartStatePatch["activeTab"],
): StudioAnalysisStartStatePatch {
  return activeTab === undefined
    ? { error: null, isSimulating: true }
    : { activeTab, error: null, isSimulating: true };
}

/**
 * Report a failed analysis and end the run.
 *
 * An `Error` with a message contributes its message; anything else is
 * stringified, because a rejection is not required to be an `Error` and
 * showing nothing would be worse than showing its text.
 *
 * @param error - Whatever was thrown or rejected.
 * @returns The patch.
 */
export function studioAnalysisFailureState(error: unknown): StudioAnalysisFailureStatePatch {
  return {
    error: error instanceof Error && error.message.length > 0 ? error.message : String(error),
    isSimulating: false,
  };
}

/**
 * Show a message without claiming the run ended.
 *
 * @param error - The message to show.
 * @returns The patch.
 */
export function studioAnalysisErrorState(error: string): StudioAnalysisErrorStatePatch {
  return { error };
}

/**
 * Mark nothing as in flight, leaving results and errors alone.
 *
 * @returns The patch.
 */
export function studioAnalysisIdleState(): StudioAnalysisIdleStatePatch {
  return { isSimulating: false };
}

/**
 * Take a finished run into the store.
 *
 * @param result - The run.
 * @returns The patch.
 */
export function studioSimulationResultState(
  result: SimulateResponse,
): StudioSimulationResultStatePatch {
  return { isSimulating: false, result };
}

/**
 * Take a finished f-I sweep into the store.
 *
 * @param fiResult - The sweep.
 * @returns The patch.
 */
export function studioFICurveResultState(
  fiResult: FICurveResponse,
): StudioFICurveResultStatePatch {
  return { fiResult, isSimulating: false };
}

/**
 * Take a finished attractor sweep into the store.
 *
 * @param bifResult - The sweep.
 * @returns The patch.
 */
export function studioBifurcationResultState(
  bifResult: BifurcationResponse,
): StudioBifurcationResultStatePatch {
  return { bifResult, isSimulating: false };
}

/**
 * Take a finished elasticity analysis into the store.
 *
 * @param sensResult - The analysis.
 * @returns The patch.
 */
export function studioSensitivityResultState(
  sensResult: SensitivityResponse,
): StudioSensitivityResultStatePatch {
  return { isSimulating: false, sensResult };
}

/**
 * Take a finished precision comparison into the store.
 *
 * @param precResult - The comparison.
 * @returns The patch.
 */
export function studioPrecisionResultState(
  precResult: PrecisionResponse,
): StudioPrecisionResultStatePatch {
  return { isSimulating: false, precResult };
}

/**
 * Take a finished two-parameter sweep into the store.
 *
 * @param heatmapResult - The sweep.
 * @returns The patch.
 */
export function studioHeatmapResultState(
  heatmapResult: HeatmapResponse,
): StudioHeatmapResultStatePatch {
  return { heatmapResult, isSimulating: false };
}

/**
 * Open the code tab before the generated script arrives.
 *
 * @returns The patch.
 */
export function studioCodegenStartState(): StudioCodegenStartStatePatch {
  return { activeTab: "code" };
}

/**
 * Take a generated script into the store.
 *
 * The digest travels with the script because the script checks it before
 * reporting a result; without it a replay could agree by coincidence.
 *
 * @param codeScript - The script.
 * @param codeOneliner - Its one-line form.
 * @param codeReplayScript - Its replay form.
 * @param codeExperimentSha256 - The experiment the script is pinned to.
 * @returns The patch.
 */
export function studioCodegenResultState(
  codeScript: string,
  codeOneliner: string,
  codeReplayScript: string,
  codeExperimentSha256: string,
): StudioCodegenResultStatePatch {
  return { codeExperimentSha256, codeOneliner, codeReplayScript, codeScript };
}

/**
 * Take a finished multi-model overlay into the store.
 *
 * @param multiResults - The runs, in the order to draw them.
 * @returns The patch.
 */
export function studioMultiResultsState(
  multiResults: SimulateResponse[],
): StudioMultiResultsStatePatch {
  return { isSimulating: false, multiResults };
}

/**
 * Take a finished network run into the store.
 *
 * @param networkResult - The run.
 * @returns The patch.
 */
export function studioNetworkResultState(
  networkResult: NetworkResult,
): StudioNetworkResultStatePatch {
  return { isSimulating: false, networkResult };
}

/**
 * Take a recorded trace into the store and show it.
 *
 * @param importedTrace - The trace.
 * @returns The patch.
 */
export function studioImportedTraceState(
  importedTrace: ImportedTrace,
): StudioImportedTraceStatePatch {
  return { activeTab: "trace", importedTrace };
}

/**
 * Take a finished two-run comparison into the store.
 *
 * @param compareResult - The comparison.
 * @returns The patch.
 */
export function studioCompareResultState(
  compareResult: CompareResponse,
): StudioCompareResultStatePatch {
  return { compareResult, isSimulating: false };
}

/**
 * Take finished nullclines into the store and show the phase portrait.
 *
 * @param nullclineResult - The nullclines.
 * @returns The patch.
 */
export function studioNullclineResultState(
  nullclineResult: NullclineResponse,
): StudioNullclineResultStatePatch {
  return { activeTab: "phase", isSimulating: false, nullclineResult };
}

/**
 * Take a finished frequency sweep into the store.
 *
 * @param freqResult - The sweep.
 * @returns The patch.
 */
export function studioFrequencyResultState(
  freqResult: FreqResponse,
): StudioFrequencyResultStatePatch {
  return { freqResult, isSimulating: false };
}

/**
 * Compute a spike-triggered average, or say there were too few spikes.
 *
 * The average is taken over the full-resolution trace, never the display
 * projection: spike indices are raw steps, and averaging on the display clock
 * would sample the wrong instants.
 *
 * @param result - The run to average.
 * @returns The patch, or `null` when there were fewer than three spikes.
 */
export function studioSTAResultState(result: SimulateResponse): StudioSTAResultStatePatch | null {
  if (result.spikes.length < 3) {
    return null;
  }
  // Spike indices are raw steps, so the average must be taken over the
  // full-resolution trace, never over the display projection.
  const variables = fullStateNames(result);
  const firstVariable = variables[0];
  if (firstVariable === undefined) {
    return null;
  }
  const voltage = fullStateTrace(result, firstVariable);
  if (voltage === undefined) {
    return null;
  }
  const halfWin = Math.min(Math.floor(10 / result.dt), 200);
  const snippets: number[][] = [];
  for (const index of result.spikes) {
    if (index - halfWin >= 0 && index + halfWin < voltage.length) {
      snippets.push(voltage.slice(index - halfWin, index + halfWin));
    }
  }
  if (snippets.length === 0) {
    return null;
  }
  const average = at(snippets, 0).map((_, index) =>
    snippets.reduce((sum, snippet) => sum + at(snippet, index), 0) / snippets.length,
  );
  const timeMs = average.map((_, index) => (index - halfWin) * result.dt);
  return {
    activeTab: "sta",
    staResult: {
      average,
      n_spikes: snippets.length,
      time_ms: timeMs,
    },
  };
}
