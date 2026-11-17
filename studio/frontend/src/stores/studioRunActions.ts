// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio simulation and analysis run actions

import { runStoreSimulation } from "./studioSimulation";
import { studioNullclineRanges } from "../studioNullclineRanges";
import {
  fetchPrecision,
  fetchCompare,
  fetchNullclines,
  fetchFreqResponse,
  fetchMultiSimulate,
  importTrace,
  simulateNetwork,
} from "../api/client";
import { studioTraceImportRequest } from "../studioTraceImport";
import {
  studioFrequencyResponseRequest,
  studioNullclineRequest,
  studioPrecisionRequest,
  studioSimulationConfig,
} from "../studioSimulationConfig";
import {
  studioAnalysisErrorState,
  studioAnalysisFailureState,
  studioAnalysisStartState,
  studioCompareResultState,
  studioFrequencyResultState,
  studioImportedTraceState,
  studioMultiResultsState,
  studioNetworkResultState,
  studioNullclineResultState,
  studioPrecisionResultState,
} from "../studioAnalysisState";
import { computeStudioSTA } from "./studioTraceActions";
import { runStoreCharacterize } from "./studioCharacterize";
import { networkParamState } from "../studioInputState";
import { runStoreDirectAnalysis } from "./studioDirectAnalysis";
import { runStoreHeavyAnalysis, simulationConfigInput } from "./studioHeavyAnalysis";
import { scheduleStudioAutoSimulation, type StudioAutoSimulationTimer } from "../studioAutoSimulation";
import { studioPrecisionKey } from "../studioExperimentKey";
import { studioStageSet } from "./studioStageFailure";
import type { StudioState, StudioStoreActions } from "./studioTypes";

/**
 * The pending auto-simulation, if one is scheduled.
 *
 * Module-level rather than in the store because it is a timer handle, not
 * state: nothing renders from it, and putting it in the store would make every
 * keystroke that reschedules a run also re-render every subscriber.
 */
let debounceTimer: StudioAutoSimulationTimer | null = null;

/**
 * The run actions: simulating the current model, and simulating it again shortly after each edit,
 * and the analyses built on simulation: the f-I curve, bifurcation, sensitivity, fixed-point
 * precision, the parameter heatmap, characterisation, several models side by side, the network run
 * and its parameters, an imported CSV trace, a comparison with a second configuration, nullclines,
 * the frequency response and the spike-triggered average.
 */
export type StudioRunActions = Pick<StudioStoreActions,
  "autoSimulate" | "runSimulation" | "runFICurve" | "runBifurcation" | "runSensitivity" | "runPrecision" | "runHeatmap" | "runCharacterize" | "runMultiSimulate" | "setNetworkParam" | "runNetwork" | "importCSV" | "runCompare" | "runNullclines" | "runFreqResponse" | "computeSTA">;

/**
 * Bind these actions to the Studio store.
 *
 * The run actions: simulating the current model, and simulating it again shortly after each edit,
 * and the analyses built on simulation: the f-I curve, bifurcation, sensitivity, fixed-point
 * precision, the parameter heatmap, characterisation, several models side by side, the network run
 * and its parameters, an imported CSV trace, a comparison with a second configuration, nullclines,
 * the frequency response and the spike-triggered average. Each run goes through the stage it
 * belongs to, so a refusal or an outdated reply is handled there.
 *
 * @param set - Writes a patch into the store.
 * @param get - Reads the store as it stands.
 * @returns The actions.
 */
export function createStudioRunActions(
  set: (partial: Partial<StudioState> | ((state: StudioState) => Partial<StudioState>)) => void,
  get: () => StudioState,
): StudioRunActions {
  return {
  autoSimulate: () => {
    debounceTimer = scheduleStudioAutoSimulation(debounceTimer, () => {
      void get().runSimulation();
    });
  },

  runSimulation: () => runStoreSimulation(get, studioStageSet("simulate", get, set)),

  runFICurve: () => runStoreHeavyAnalysis("fi_curve", get, studioStageSet("analyse", get, set)),

  runBifurcation: () => runStoreHeavyAnalysis("bifurcation", get, studioStageSet("analyse", get, set)),

  runSensitivity: () => runStoreHeavyAnalysis("sensitivity", get, studioStageSet("analyse", get, set)),

  runPrecision: () => runStoreDirectAnalysis(get, studioStageSet("analyse", get, set), async (s) => {
    if (s.sourceMode !== "ode") {
      throw new Error("Precision compare only for custom ODE mode");
    }
    const precResult = await fetchPrecision(
      studioPrecisionRequest(simulationConfigInput(s), s.modelQFormat),
    );
    return studioPrecisionResultState(precResult);
  }, "precision", (s) => studioPrecisionKey(simulationConfigInput(s), s.modelQFormat)),

  runHeatmap: () => runStoreHeavyAnalysis("heatmap", get, studioStageSet("analyse", get, set)),

  runCharacterize: () => {
    runStoreCharacterize(get, studioStageSet("analyse", get, set));
  },

  runMultiSimulate: async (modelNames) => {
    const s = get();
    if (s.isSimulating) return;
    set(studioAnalysisStartState("multi"));
    try {
      const configs = modelNames.slice(0, 4).map((name) => ({
        name, params: null, dt: null, duration: s.duration, current: s.current, protocol: s.protocol,
      }));
      const multiResults = await fetchMultiSimulate(configs);
      set(studioMultiResultsState(multiResults));
    } catch (e) { set(studioAnalysisFailureState(e)); }
  },

  setNetworkParam: (key, value) => {
    set((s) => networkParamState(s.networkParams, key, value));
  },

  runNetwork: async () => {
    const s = get();
    if (s.isSimulating) return;
    set(studioAnalysisStartState("network"));
    try {
      const np = s.networkParams;
      const networkResult = await simulateNetwork({
        ...np, duration: s.duration,
      });
      set(studioNetworkResultState(networkResult));
    } catch (e) { set(studioAnalysisFailureState(e)); }
  },

  importCSV: async (csv) => {
    try {
      const importedTrace = await importTrace(studioTraceImportRequest(csv, get().dt));
      set(studioImportedTraceState(importedTrace));
    } catch (e) { set(studioAnalysisErrorState(e instanceof Error ? e.message : String(e))); }
  },

  runCompare: (configB) => runStoreDirectAnalysis(get, studioStageSet("analyse", get, set), async (s) => {
    const configA = studioSimulationConfig(simulationConfigInput(s));
    const compareResult = await fetchCompare(configA, configB);
    return studioCompareResultState(compareResult);
  }, "compare"),

  runNullclines: () => runStoreDirectAnalysis(get, studioStageSet("analyse", get, set), async (s) => {
    if (s.sourceMode !== "ode" || s.equations.length < 2) {
      throw new Error("Nullclines need 2+ variable ODE in custom mode");
    }
    const vars = Object.keys(s.odeInit);
    const [var0, var1] = vars;
    if (var0 === undefined || var1 === undefined) throw new Error("Nullclines need initial values for two variables");
    const ranges = studioNullclineRanges(s, var0, var1);
    const nullclineResult = await fetchNullclines(
      studioNullclineRequest({
        equations: s.equations,
        odeParams: s.odeParams,
        odeInit: s.odeInit,
        protocol: s.protocol,
        current: s.current,
        ranges,
        gridSize: 60,
      }),
    );
    return studioNullclineResultState(nullclineResult);
  }, "phase"),

  runFreqResponse: () => runStoreDirectAnalysis(get, studioStageSet("analyse", get, set), async (s) => {
    const cfg = studioSimulationConfig(simulationConfigInput(s));
    const freqResult = await fetchFreqResponse(studioFrequencyResponseRequest(cfg, s.current));
    return studioFrequencyResultState(freqResult);
  }, "freq"),

  computeSTA: () => { computeStudioSTA(get, set); },
  };
}
