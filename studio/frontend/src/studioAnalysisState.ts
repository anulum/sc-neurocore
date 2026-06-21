// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio analysis store state helpers
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

export interface StudioAnalysisStartStatePatch {
  activeTab?: "bifurcation" | "compare" | "fi-curve" | "freq" | "heatmap" | "multi" |
    "network" | "precision" | "sensitivity";
  error: null;
  isSimulating: true;
}

export interface StudioAnalysisFailureStatePatch {
  error: string;
  isSimulating: false;
}

export interface StudioAnalysisErrorStatePatch {
  error: string;
}

export interface StudioAnalysisIdleStatePatch {
  isSimulating: false;
}

export interface StudioSimulationResultStatePatch {
  isSimulating: false;
  result: SimulateResponse;
}

export interface StudioFICurveResultStatePatch {
  fiResult: FICurveResponse;
  isSimulating: false;
}

export interface StudioBifurcationResultStatePatch {
  bifResult: BifurcationResponse;
  isSimulating: false;
}

export interface StudioSensitivityResultStatePatch {
  isSimulating: false;
  sensResult: SensitivityResponse;
}

export interface StudioPrecisionResultStatePatch {
  isSimulating: false;
  precResult: PrecisionResponse;
}

export interface StudioHeatmapResultStatePatch {
  heatmapResult: HeatmapResponse;
  isSimulating: false;
}

export interface StudioCodegenStartStatePatch {
  activeTab: "code";
}

export interface StudioCodegenResultStatePatch {
  codeOneliner: string;
  codeScript: string;
}

export interface StudioMultiResultsStatePatch {
  isSimulating: false;
  multiResults: SimulateResponse[];
}

export interface StudioNetworkResultStatePatch {
  isSimulating: false;
  networkResult: NetworkResult;
}

export interface StudioImportedTraceStatePatch {
  activeTab: "trace";
  importedTrace: ImportedTrace;
}

export interface StudioCompareResultStatePatch {
  compareResult: CompareResponse;
  isSimulating: false;
}

export interface StudioNullclineResultStatePatch {
  activeTab: "phase";
  isSimulating: false;
  nullclineResult: NullclineResponse;
}

export interface StudioFrequencyResultStatePatch {
  freqResult: FreqResponse;
  isSimulating: false;
}

export interface StudioSTAResultStatePatch {
  activeTab: "sta";
  staResult: {
    average: number[];
    n_spikes: number;
    time_ms: number[];
  };
}

export function studioAnalysisStartState(
  activeTab?: StudioAnalysisStartStatePatch["activeTab"],
): StudioAnalysisStartStatePatch {
  return activeTab === undefined
    ? { error: null, isSimulating: true }
    : { activeTab, error: null, isSimulating: true };
}

export function studioAnalysisFailureState(error: unknown): StudioAnalysisFailureStatePatch {
  return {
    error: error instanceof Error && error.message.length > 0 ? error.message : String(error),
    isSimulating: false,
  };
}

export function studioAnalysisErrorState(error: string): StudioAnalysisErrorStatePatch {
  return { error };
}

export function studioAnalysisIdleState(): StudioAnalysisIdleStatePatch {
  return { isSimulating: false };
}

export function studioSimulationResultState(
  result: SimulateResponse,
): StudioSimulationResultStatePatch {
  return { isSimulating: false, result };
}

export function studioFICurveResultState(
  fiResult: FICurveResponse,
): StudioFICurveResultStatePatch {
  return { fiResult, isSimulating: false };
}

export function studioBifurcationResultState(
  bifResult: BifurcationResponse,
): StudioBifurcationResultStatePatch {
  return { bifResult, isSimulating: false };
}

export function studioSensitivityResultState(
  sensResult: SensitivityResponse,
): StudioSensitivityResultStatePatch {
  return { isSimulating: false, sensResult };
}

export function studioPrecisionResultState(
  precResult: PrecisionResponse,
): StudioPrecisionResultStatePatch {
  return { isSimulating: false, precResult };
}

export function studioHeatmapResultState(
  heatmapResult: HeatmapResponse,
): StudioHeatmapResultStatePatch {
  return { heatmapResult, isSimulating: false };
}

export function studioCodegenStartState(): StudioCodegenStartStatePatch {
  return { activeTab: "code" };
}

export function studioCodegenResultState(
  codeScript: string,
  codeOneliner: string,
): StudioCodegenResultStatePatch {
  return { codeOneliner, codeScript };
}

export function studioMultiResultsState(
  multiResults: SimulateResponse[],
): StudioMultiResultsStatePatch {
  return { isSimulating: false, multiResults };
}

export function studioNetworkResultState(
  networkResult: NetworkResult,
): StudioNetworkResultStatePatch {
  return { isSimulating: false, networkResult };
}

export function studioImportedTraceState(
  importedTrace: ImportedTrace,
): StudioImportedTraceStatePatch {
  return { activeTab: "trace", importedTrace };
}

export function studioCompareResultState(
  compareResult: CompareResponse,
): StudioCompareResultStatePatch {
  return { compareResult, isSimulating: false };
}

export function studioNullclineResultState(
  nullclineResult: NullclineResponse,
): StudioNullclineResultStatePatch {
  return { activeTab: "phase", isSimulating: false, nullclineResult };
}

export function studioFrequencyResultState(
  freqResult: FreqResponse,
): StudioFrequencyResultStatePatch {
  return { freqResult, isSimulating: false };
}

export function studioSTAResultState(result: SimulateResponse): StudioSTAResultStatePatch | null {
  if (result.spikes.length < 3) {
    return null;
  }
  const variables = Object.keys(result.states);
  const voltage = result.states[variables[0]];
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
  const average = snippets[0].map((_, index) =>
    snippets.reduce((sum, snippet) => sum + snippet[index], 0) / snippets.length,
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
