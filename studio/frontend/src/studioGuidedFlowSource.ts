// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — The store fields the guided workflow reads

/**
 * What the guided workflow reads, declared apart from the store that holds it.
 *
 * Naming the fields rather than taking the whole state does two things: a case
 * can stand one of these up without assembling a store, and a field added to
 * the store does not silently become an input to the workflow's judgement of
 * what is done.
 */

import type {
  BifurcationResponse,
  CharacterizeResponse,
  CompileTraceability,
  CompareResponse,
  FICurveResponse,
  FreqResponse,
  HeatmapResponse,
  ModelCosimReport,
  MultiTargetResult,
  NullclineResponse,
  PrecisionResponse,
  SensitivityResponse,
  SimulateResponse,
  StudioEvidenceBundleResponse,
  SynthResult,
} from "./api/client";
import type { StudioProjectTrainingConfig } from "./studioProjectState";
import type { StudioSimulationConfigSource } from "./studioSimulationConfigInput";

/** Every field the guided workflow's completion rules read. */
export interface StudioGuidedFlowSource extends StudioSimulationConfigSource {
  analysisExperimentKey: string | null;
  modelQFormat: string;
  bifResult: BifurcationResponse | null;
  charResult: CharacterizeResponse | null;
  compareResult: CompareResponse | null;
  compileEvidenceBundle: StudioEvidenceBundleResponse | null;
  compileTraceability: CompileTraceability | null;
  cosimResult: ModelCosimReport | null;
  evidenceBundle: StudioEvidenceBundleResponse | null;
  fiResult: FICurveResponse | null;
  freqResult: FreqResponse | null;
  heatmapResult: HeatmapResponse | null;
  multiTargetResult: MultiTargetResult | null;
  nullclineResult: NullclineResponse | null;
  precResult: PrecisionResponse | null;
  projectEvidenceBundle: StudioEvidenceBundleResponse | null;
  result: SimulateResponse | null;
  resultExperimentKey: string | null;
  sensResult: SensitivityResponse | null;
  synthesisEvidenceBundle: StudioEvidenceBundleResponse | null;
  synthResult: SynthResult | null;
  trainingConfig: StudioProjectTrainingConfig;
  trainingExperimentKey: string | null;
  trainingStatus: string;
}
