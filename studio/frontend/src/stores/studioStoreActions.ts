// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio Zustand store
// Studio store action implementations (mutations + async side effects).

import { createStudioGraphActions } from "./studioGraphActions";
import { createStudioHardwareActions } from "./studioHardwareActions";
import { createStudioSessionAndShareActions } from "./studioSessionAndShareActions";
import { createStudioProjectActions } from "./studioProjectActions";
import { createStudioAuthActions } from "./studioAuthActions";
import { createStudioAdminActions } from "./studioAdminActions";
import { createStudioTrainingActions } from "./studioTrainingActions";
import { runStoreSimulation } from "./studioSimulation";
import { studioNullclineRanges } from "../studioNullclineRanges";
import {
  fetchTemplates,
  fetchModels,
  fetchModelDetail,
  fetchPresets,
  fetchPreset,
  fetchStudioCapabilities,
  fetchPrecision,
  fetchCodegen,
  fetchReplayNotebook,
  fetchReplayPack,
  fetchCompare,
  fetchNullclines,
  fetchFreqResponse,
  fetchMultiSimulate,
  importTrace,
  simulateNetwork,
  runPipeline as apiRunPipeline,
} from "../api/client";
import {
  studioProjectFailureState,
} from "../studioProjectState";
import {
  studioGraphFailureState,
  studioGraphRequest,
  studioPipelineCompletedState,
  studioPipelineStartState,
} from "../studioGraphRequests";
import {
  studioTraceImportRequest,
} from "../studioTraceImport";
import {
  studioExperimentExportRequest,
  studioFrequencyResponseRequest,
  studioNullclineRequest,
  studioPrecisionRequest,
  studioSimulationConfig,
} from "../studioSimulationConfig";
import {
  studioAnalysisErrorState,
  studioAnalysisFailureState,
  studioAnalysisStartState,
  studioCodegenResultState,
  studioCodegenStartState,
  studioCompareResultState,
  studioFrequencyResultState,
  studioImportedTraceState,
  studioMultiResultsState,
  studioNetworkResultState,
  studioNullclineResultState,
  studioPrecisionResultState,
} from "../studioAnalysisState";
import {
  replayPackExport,
} from "../simulationExports";
import { computeStudioSTA, exportStudioTrace } from "./studioTraceActions";
import { downloadBrowserArtefact } from "../browserArtefactDownload";

import {
} from "../evidenceBundles";
import {
  capabilityFailureState,
  capabilityLoadedState,
  capabilityLoadingState,
} from "../capabilityShell";
import {
  compilerConfigurationInvalidatedState,
  compilerCosimInvalidatedState,
} from "../compilerStoreState";
import {
  modelDetailLoadedState,
  modelSelectionStartedState,
  modelsLoadedState,
  presetSelection,
  presetsLoadedState,
  templateSelectedState,
  templatesLoadedState,
} from "../modelSelectionStoreState";
import { runStoreCharacterize } from "./studioCharacterize";
import {
  frequencyHzState,
  seedState,
  trialState,
  activeTabState,
  currentState,
  dtState,
  durationState,
  equationsState,
  modelDefaultsState,
  modelFilterState,
  networkParamState,
  numberRecordEntryState,
  protocolState,
  resetState,
  sourceModeState,
  sweepParamState,
  sweepParamYState,
  thresholdState,
} from "../studioInputState";
import type {
  StudioState,
  StudioStoreActions,
} from "./studioTypes";
import { runStoreDirectAnalysis } from "./studioDirectAnalysis";
import {
  runStoreHeavyAnalysis,
  simulationConfigInput,
} from "./studioHeavyAnalysis";
import {
  scheduleStudioAutoSimulation,
  type StudioAutoSimulationTimer,
} from "../studioAutoSimulation";
import { studioPrecisionKey } from "../studioExperimentKey";
import type { StudioEquationIrActions } from "./studioEquationIrActions";
import { studioStageOutcomeState, studioStageSet } from "./studioStageFailure";



/**
 * The pending auto-simulation, if one is scheduled.
 *
 * Module-level rather than in the store because it is a timer handle, not
 * state: nothing renders from it, and putting it in the store would make every
 * keystroke that reschedules a run also re-render every subscriber.
 */
let debounceTimer: StudioAutoSimulationTimer | null = null;

/**
 * Build every action the Studio store exposes.
 *
 * The actions are closures over `set` and `get` rather than methods, which is
 * what lets each one read the state as it stands when it runs rather than as
 * it stood when it was created -- an action that awaits a request must see the
 * store the user has since changed.
 *
 * **A run started by an action is deliberately not awaited.** Several actions
 * end by starting a simulation, and they mark it `void`: the caller changed a
 * parameter and the run that follows is a consequence, not a result. Awaiting
 * it would make every setter as slow as a simulation, and the run reports
 * itself through the store as it progresses.
 *
 * @param set - Writes a patch into the store.
 * @param get - Reads the store as it stands.
 * @returns Complete actions except native equation actions, which are composed
 *   separately. Every other callable state field remains required here.
 */
export function createStudioStoreActions(
  set: (partial: Partial<StudioState> | ((state: StudioState) => Partial<StudioState>)) => void,
  get: () => StudioState,
): Omit<StudioStoreActions, keyof StudioEquationIrActions> {
  const training = createStudioTrainingActions(set, get);
  let modelSelectionVersion = 0;
  let modelsLoadVersion = 0;
  let presetsLoadVersion = 0;

  /** Fetch restored model metadata without replacing saved inputs with catalogue defaults. */
  const runRestoredExperiment = (): void => {
    const version = ++modelSelectionVersion;
    const { sourceMode, selectedModelName } = get();
    if (sourceMode !== "model") {
      void get().runSimulation();
      return;
    }
    set({ modelDetail: null });
    void fetchModelDetail(selectedModelName).then((detail) => {
      if (version !== modelSelectionVersion) return;
      set({ modelDetail: detail });
      void get().runSimulation();
    }).catch((error: unknown) => {
      if (version === modelSelectionVersion) {
        set(studioProjectFailureState(error, "Restored model details could not be loaded"));
      }
    });
  };

  return {
  setSourceMode: (m) => {
    const changed = get().sourceMode !== m;
    if (m === "ode") modelSelectionVersion += 1;
    set({ ...sourceModeState(m), ...compilerConfigurationInvalidatedState() });
    if (m === "model" && get().modelDetail === null && get().selectedModelName) {
      void get().selectModel(get().selectedModelName);
    } else if (changed) {
      // The run on screen belongs to the other source. Switching to the ODE
      // editor kept showing the catalogue model's trace, spike statistics
      // and firing pattern beside the equations until something was edited.
      get().autoSimulate();
    }
  },
  setEquations: (eqs) => {
    set({ ...equationsState(eqs), ...compilerConfigurationInvalidatedState() });
    get().autoSimulate();
  },
  setThreshold: (t) => {
    set({ ...thresholdState(t), ...compilerConfigurationInvalidatedState() });
    get().autoSimulate();
  },
  setReset: (r) => {
    set({ ...resetState(r), ...compilerConfigurationInvalidatedState() });
    get().autoSimulate();
  },
  setOdeParam: (key, value) => {
    set((s) => ({
      ...numberRecordEntryState("odeParams", s.odeParams, key, value),
      ...compilerConfigurationInvalidatedState(),
    }));
    get().autoSimulate();
  },
  setOdeInit: (key, value) => {
    set((s) => ({
      ...numberRecordEntryState("odeInit", s.odeInit, key, value),
      ...compilerConfigurationInvalidatedState(),
    }));
    get().autoSimulate();
  },
  setModelParam: (key, value) => {
    set((s) => ({
      ...numberRecordEntryState("modelParams", s.modelParams, key, value),
      ...compilerConfigurationInvalidatedState(),
    }));
    get().autoSimulate();
  },
  setModelIntegrator: (modelIntegrator) => { set({
    modelIntegrator,
    ...compilerConfigurationInvalidatedState(),
  }); },
  setModelQFormat: (modelQFormat) => { set({
    modelQFormat,
    ...compilerConfigurationInvalidatedState(),
  }); },
  setDt: (dt) => {
    set({ ...dtState(dt), ...compilerConfigurationInvalidatedState() });
    get().autoSimulate();
  },
  setDuration: (d) => { set(durationState(d)); get().autoSimulate(); },
  setCurrent: (c) => {
    set({ ...currentState(c), ...compilerCosimInvalidatedState(get().sourceMode === "model") });
    get().autoSimulate();
  },
  setProtocol: (p) => { set(protocolState(p)); get().autoSimulate(); },
  setFrequencyHz: (frequencyHz) => { set(frequencyHzState(frequencyHz)); get().autoSimulate(); },
  setSeed: (seed) => { set(seedState(seed, get().sourceMode === "model" ? get().modelParams : undefined)); get().autoSimulate(); },
  setTrial: (trial) => { set(trialState(trial)); get().autoSimulate(); },
  setActiveTab: (tab) => { set(activeTabState(tab)); },
  setCandidateDraft: (text) => { set({ candidates: text === "" ? [] : [{ text }] }); },
  setModelFilter: (f) => { set(modelFilterState(f)); },
  setSweepParam: (p) => { set(sweepParamState(p)); },
  setSweepParamY: (p) => { set(sweepParamYState(p)); },

  loadTemplates: async () => { set(templatesLoadedState(await fetchTemplates())); },
  loadCapabilities: async () => {
    set(capabilityLoadingState());
    try {
      const response = await fetchStudioCapabilities();
      set(capabilityLoadedState(response.capabilities));
    } catch (error: unknown) {
      set(capabilityFailureState(error));
    }
  },
  ...createStudioAuthActions(set, get, training),
  ...createStudioAdminActions(set, get),
  loadModels: async () => {
    const version = ++modelsLoadVersion;
    let selectionVersion: number | null = null;
    try {
      const models = await fetchModels();
      if (version !== modelsLoadVersion) return;
      const state = get();
      set({ ...modelsLoadedState(models), modelsError: null });
      const firstModel = models[0];
      if (firstModel !== undefined && state.sourceMode === "model"
        && (!state.selectedModelName || (state.modelsError !== null && state.modelDetail === null))) {
        const selection = get().selectModel(state.selectedModelName || firstModel.name);
        selectionVersion = modelSelectionVersion;
        await selection;
      }
    } catch (error: unknown) {
      if (selectionVersion === null
        ? version === modelsLoadVersion : selectionVersion === modelSelectionVersion) {
        set({ modelsError: error instanceof Error ? error.message : "Model library initialization failed" });
      }
    }
  },
  loadPresets: async () => {
    const version = ++presetsLoadVersion;
    try {
      const presets = await fetchPresets();
      if (version === presetsLoadVersion) set({ ...presetsLoadedState(presets), presetsError: null });
    } catch (error: unknown) {
      if (version === presetsLoadVersion) {
        set({ presetsError: error instanceof Error ? error.message : "Experiment library load failed" });
      }
    }
  },

  selectTemplate: (name) => {
    const template = get().templates.find((candidate) => candidate.name === name);
    if (template === undefined) return;
    modelSelectionVersion += 1;
    set({ ...templateSelectedState(template), ...compilerConfigurationInvalidatedState() });
    void get().runSimulation();
  },

  selectModel: async (name) => {
    const version = ++modelSelectionVersion;
    set({ ...modelSelectionStartedState(name), ...compilerConfigurationInvalidatedState() });
    // No `detail === null` guard: the route's contract does not allow one, and
    // the guard that used to be here defended a single field of a response that
    // nothing validates. Response validation at the API boundary is recorded as
    // its own unit.
    const detail = await fetchModelDetail(name);
    // An older request may finish after a newer model has been selected.
    if (version !== modelSelectionVersion) return;
    set(modelDetailLoadedState(detail));
    void get().runSimulation();
  },

  loadPreset: async (id) => {
    const preset = await fetchPreset(id);
    const selection = presetSelection(preset);
    if (selection.modelName !== null) {
      await get().selectModel(selection.modelName);
      if (selection.modelRuntimeState !== null) set(selection.modelRuntimeState);
    } else if (selection.odeState !== null) {
      set({ ...selection.odeState, ...compilerConfigurationInvalidatedState() });
    }
    if (selection.action.kind === "fi-curve") void get().runFICurve();
    else if (selection.action.kind === "precision") void get().runPrecision();
    else {
      set(activeTabState(selection.action.activeTab));
      void get().runSimulation();
    }
  },

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

  ...createStudioHardwareActions(set, get),

  recordStageOutcome: (stage, failure) => {
    set(studioStageOutcomeState(stage, failure, get()));
  },

  runHeatmap: () => runStoreHeavyAnalysis("heatmap", get, studioStageSet("analyse", get, set)),

  runCodegen: async () => {
    const s = get();
    set(studioCodegenStartState());
    try {
      if (s.sourceMode === "model" && s.modelDetail?.name !== s.selectedModelName) {
        throw new Error("Selected model is still loading");
      }
      const res = await fetchCodegen(studioExperimentExportRequest(simulationConfigInput(s)));
      set(studioCodegenResultState(res.script, res.oneliner, res.replay_script, res.experiment_sha256));
    } catch (e) { set(studioAnalysisErrorState(e instanceof Error ? e.message : String(e))); }
  },

  exportReplayPack: async () => {
    const s = get();
    try {
      if (s.sourceMode === "model" && s.modelDetail?.name !== s.selectedModelName) {
        throw new Error("Selected model is still loading");
      }
      const pack = await fetchReplayPack(studioExperimentExportRequest(simulationConfigInput(s)));
      const artefact = replayPackExport(pack);
      downloadBrowserArtefact(artefact.blob, artefact.filename);
    } catch (e) { set(studioAnalysisErrorState(e instanceof Error ? e.message : String(e))); }
  },

  exportReplayNotebook: async () => {
    const s = get();
    try {
      if (s.sourceMode === "model" && s.modelDetail?.name !== s.selectedModelName) {
        throw new Error("Selected model is still loading");
      }
      const notebook = await fetchReplayNotebook(studioExperimentExportRequest(simulationConfigInput(s)));
      const stem = s.sourceMode === "model" ? s.selectedModelName : "equations";
      downloadBrowserArtefact(
        new Blob([JSON.stringify(notebook, null, 1)], { type: "application/x-ipynb+json" }),
        `${stem}-replay.ipynb`,
      );
    } catch (e) { set(studioAnalysisErrorState(e instanceof Error ? e.message : String(e))); }
  },

  exportData: () => { exportStudioTrace("json", get, set); },
  exportCSV: () => { exportStudioTrace("csv", get, set); },
  exportSVG: () => { exportStudioTrace("svg", get, set); },

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


  ...createStudioProjectActions(set, get, runRestoredExperiment),

  runPipelineAction: async () => {
    const s = get();
    if (s.isSimulating || s.graphPopulations.length === 0) return;
    set(studioPipelineStartState());
    try {
      const graph = studioGraphRequest(s.graphPopulations, s.graphProjections, s.duration, s.dt, s.seed);
      const pipelineResult = await apiRunPipeline(graph, s.synthTarget);
      set(studioPipelineCompletedState(pipelineResult));
    } catch (e) {
      set({ ...studioGraphFailureState(e, "Pipeline run failed", { clearBusy: true }), pipelineRunning: false });
    }
  },

  ...createStudioGraphActions(set, get),

  ...training.actions,

  resetDefaults: () => {
    const s = get();
    if (s.sourceMode === "model" && s.modelDetail) {
      set({ ...modelDefaultsState(s.modelDetail), ...compilerConfigurationInvalidatedState() });
    }
    void get().runSimulation();
  },

  ...createStudioSessionAndShareActions(set, get, runRestoredExperiment),
  };
}
