// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio Zustand store
// Studio store action implementations (mutations + async side effects).

import { createStudioGraphActions } from "./studioGraphActions";
import { createStudioAdminActions } from "./studioAdminActions";
import { createStudioTrainingActions } from "./studioTrainingActions";
import { runStoreSimulation } from "./studioSimulation";
import { studioNullclineRanges } from "../studioNullclineRanges";
import { readStudioModelLink, readStudioStartupHashState } from "../studioStartupRuntime";
import { studioModelLinkDecision, studioShareLinkDecision } from "../shareLinkApplication";
import {
  fetchTemplates,
  fetchModels,
  fetchModelDetail,
  fetchPresets,
  fetchPreset,
  fetchStudioAuthSession,
  fetchStudioCapabilities,
  fetchStudioJobs,
  fetchStudioOperatorStatus,
  loginStudioBrowserUser,
  logoutStudioBrowserUser,
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
  fetchSynthTools,
  runSynthesis as apiRunSynthesis,
  runSynthesisTerminal as apiRunSynthesisTerminal,
  runMultiTargetSynthesis,
  fetchSynthEstimate,
  branchRefusedEdit as apiBranchRefusedEdit,
  saveProject as apiSaveProject,
  loadProject as apiLoadProject,
  listDeletedProjects as apiListDeletedProjects,
  restoreProject as apiRestoreProject,
  listProjects as apiListProjects,
  deleteProject as apiDeleteProject,
  runPipeline as apiRunPipeline,
  setStudioAuthToken,
} from "../api/client";
import {
  clearStoredStudioAuthToken,
  studioAuthFailureState,
  studioAuthLoadingState,
  studioAuthLogoutCompleteState,
  studioAuthLogoutFailureState,
  studioAuthSessionLoadedState,
  studioAuthUnauthenticatedState,
  storeStudioAuthToken,
  syncStoredStudioAuthToken,
} from "../studioAuthSession";
import {
  studioSavedSessionRemovedState,
  studioSavedSessionRestoreState,
  captureAndStoreStudioSession,
  writeStoredStudioSessions,
} from "../studioSavedSessions";
import {
  studioProjectSaveState,
  studioProjectDeletedListedState,
  studioProjectExpectedRevision,
  studioProjectFailureState,
  studioProjectListLoadedState,
  studioProjectRestoreState,
  studioProjectRevisionFromLoadResponse,
  studioProjectSaveFailureState,
  studioProjectSavedState,
  studioProjectStateFromLoadResponse,
} from "../studioProjectState";
import {
  studioGraphFailureState,
  studioGraphRequest,
  studioPipelineCompletedState,
  studioPipelineStartState,
} from "../studioGraphRequests";
import {
  copyStudioShareUrlInRuntime,
  scheduleStudioShareStatusClear,
  studioShareStatusClearedState,
  studioShareStatusState,
} from "../studioShareRuntime";
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
  multiTargetSynthesisRunCompletedState,
  synthesisErrorMessageState,
  synthesisErrorState,
  synthesisEstimateLoadedState,
  synthesisRunCompletedState,
  synthesisTargetState,
  synthesisToolStatusLoadedState,
} from "../synthesisStoreState";
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
import { runStoreCompile } from "./studioCompile";
import type { StudioEquationIrActions } from "./studioEquationIrActions";
import { runStoreSynthesis } from "./studioSynthesis";
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
  loadAuthSession: async () => {
    const currentToken = syncStoredStudioAuthToken(setStudioAuthToken);
    if (currentToken === null) {
      set(studioAuthUnauthenticatedState());
      return;
    }
    set(studioAuthLoadingState());
    try {
      const authSession = await fetchStudioAuthSession();
      set(studioAuthSessionLoadedState(authSession));
    } catch (error: unknown) {
      clearStoredStudioAuthToken();
      syncStoredStudioAuthToken(setStudioAuthToken);
      set(studioAuthFailureState(error, "Session check failed"));
    }
  },
  loginBrowserUser: async (username, password) => {
    set(studioAuthLoadingState());
    try {
      const login = await loginStudioBrowserUser(username, password);
      storeStudioAuthToken(login.access_token);
      syncStoredStudioAuthToken(setStudioAuthToken);
      const authSession = await fetchStudioAuthSession();
      set(studioAuthSessionLoadedState(authSession));
      await get().loadOperatorStatus();
    } catch (error: unknown) {
      clearStoredStudioAuthToken();
      syncStoredStudioAuthToken(setStudioAuthToken);
      set(studioAuthFailureState(error, "Login failed"));
    }
  },
  logoutBrowserUser: async () => {
    training.releaseTrainingSession();
    set({ ...studioAuthLoadingState(), graphJobRecovery: null, graphSimResult: null });
    try {
      await logoutStudioBrowserUser();
    } catch (error: unknown) {
      set(studioAuthLogoutFailureState(error));
    } finally {
      clearStoredStudioAuthToken();
      syncStoredStudioAuthToken(setStudioAuthToken);
      set(studioAuthLogoutCompleteState());
      set({
        trainingJobs: [], trainingJobsError: null, trainingJobsLoading: false,
        trainingJobId: null, trainingStatus: "idle", trainingEpochs: [],
        trainingObservedConfig: null, trainingExperimentKey: null,
      });
    }
  },
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

  runCompile: () => runStoreCompile("compile", get, studioStageSet("compile", get, set)),

  recordStageOutcome: (stage, failure) => {
    set(studioStageOutcomeState(stage, failure, get()));
  },

  runCosim: () => runStoreCompile("cosim", get, studioStageSet("cosim", get, set)),

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


  setSynthTarget: (t) => { set(synthesisTargetState(t)); },

  runSynthesis: () => runStoreSynthesis(get, studioStageSet("synthesise", get, set), async (s) => {
      const verilog = s.svSource || s.verilogSrc;
      let resultArtifactPath = "synthesis/result.json";
      let synthResult: NonNullable<StudioState["synthResult"]>;
      if (s.sourceMode === "model") {
        if (s.compileTraceability === null || s.cosimResult === null) {
          throw new Error("Compile and bit-exact co-simulate the selected model before routing.");
        }
        if (!s.cosimResult.bit_exact
          || s.cosimResult.rtl.source_sha256 !== s.compileTraceability.output.rtl_sha256) {
          throw new Error("Selected RTL does not have current bit-exact co-simulation parity.");
        }
        if (!new Set(["ice40", "ecp5"]).has(s.synthTarget)) {
          throw new Error(`Target ${s.synthTarget} has no selected-RTL place-and-route terminal.`);
        }
        const terminal = await apiRunSynthesisTerminal(
          verilog,
          s.synthTarget,
          s.compileTraceability,
          s.cosimResult,
        );
        synthResult = { ...terminal.synthesis, silicon_terminal: terminal,
          studio_job_receipt: terminal.studio_job_receipt };
        resultArtifactPath = "synthesis/terminal-result.json";
      } else {
        synthResult = await apiRunSynthesis(verilog, s.synthTarget);
      }
      const [operatorStatus, jobList] = await Promise.all([
        fetchStudioOperatorStatus(),
        fetchStudioJobs(),
      ]);
      return synthesisRunCompletedState(
        synthResult,
        operatorStatus,
        jobList,
        resultArtifactPath,
      );
  }),

  runMultiTargetSynthesis: () => runStoreSynthesis(get, studioStageSet("synthesise", get, set), async (s) => {
    if (s.sourceMode === "model") {
      throw new Error(
        "Selected models use the digest-bound single-target synthesis/PnR terminal.",
      );
    }
      const verilog = s.svSource || s.verilogSrc;
      const multiTargetResult = await runMultiTargetSynthesis(verilog);
      const [operatorStatus, jobList] = await Promise.all([
        fetchStudioOperatorStatus(),
        fetchStudioJobs(),
      ]);
      return multiTargetSynthesisRunCompletedState(multiTargetResult, operatorStatus, jobList);
  }),

  runSynthEstimate: async () => {
    const s = get();
    const irOps = s.irText ? s.irText.split("\n").filter((l) => l.trim().startsWith("%")).length : 0;
    if (irOps < 1) {
      set(synthesisErrorMessageState("Build IR first to estimate resources"));
      return;
    }
    try {
      const synthEstimate = await fetchSynthEstimate(irOps, s.synthTarget);
      set(synthesisEstimateLoadedState(synthEstimate));
    } catch (e) { set(synthesisErrorState(e, "Synthesis estimate failed")); }
  },

  checkSynthTools: async () => {
    try {
      const toolsAvailable = await fetchSynthTools();
      set(synthesisToolStatusLoadedState(toolsAvailable));
    } catch { /* tools check is non-critical */ }
  },

  saveProjectToServer: async (name) => {
    const expected = studioProjectExpectedRevision(get().projectRevision, name);
    try {
      const state = studioProjectSaveState(get());
      const projectSaveResult = await apiSaveProject(name, state, expected);
      set(studioProjectSavedState(projectSaveResult));
      await get().listServerProjects();
    } catch (e) { set(studioProjectSaveFailureState(e, name, expected)); }
  },

  keepRefusedEdit: async () => {
    // Only reachable after a save conflict, which is the only thing that sets
    // `refusedEdit`. The state sent is the one still held here — the edit that
    // was refused — not a reload of what won.
    const pending = get().refusedEdit;
    if (!pending) return;
    try {
      const branch = await apiBranchRefusedEdit(
        pending.name,
        studioProjectSaveState(get()),
        pending.baseRevision,
      );
      set({
        error: `Kept as "${branch.branched}". Reload ${pending.name} to see the other edit.`,
        refusedEdit: null,
      });
      await get().listServerProjects();
    } catch (e) { set(studioProjectFailureState(e, "Keeping the refused edit failed")); }
  },

  loadProjectFromServer: async (name, revision = null) => {
    try {
      const data = await apiLoadProject(name, revision);
      const projectState = studioProjectStateFromLoadResponse(data, get().trainingConfig);
      set({
        ...studioProjectRestoreState(projectState),
        ...compilerConfigurationInvalidatedState(),
        projectRevision: studioProjectRevisionFromLoadResponse(data, name),
      });
      runRestoredExperiment();
    } catch (e) { set(studioProjectFailureState(e, "Project load failed")); }
  },

  listServerProjects: async () => {
    try {
      const serverProjects = await apiListProjects();
      set(studioProjectListLoadedState(serverProjects));
    } catch { /* non-critical */ }
  },

  deleteServerProject: async (name) => {
    try {
      await apiDeleteProject(name);
      await get().listServerProjects();
      // A delete is recoverable, so the trash is refreshed with the list:
      // the workspace has to be visible somewhere the moment it leaves here.
      await get().listDeletedServerProjects();
    } catch (e) { set(studioProjectFailureState(e, "Project delete failed")); }
  },

  listDeletedServerProjects: async () => {
    try {
      const { deleted } = await apiListDeletedProjects();
      set(studioProjectDeletedListedState(deleted));
    } catch (e) { set(studioProjectFailureState(e, "Deleted project list failed")); }
  },

  restoreDeletedServerProject: async (token) => {
    try {
      await apiRestoreProject(token);
      await get().listServerProjects();
      await get().listDeletedServerProjects();
    } catch (e) { set(studioProjectFailureState(e, "Project restore failed")); }
  },

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

  saveSession: (name) => {
    const s = get();
    const saved = captureAndStoreStudioSession(s, name, s.savedSessions);
    set(saved);
    return saved.sessionWriteResult;
  },

  loadSession: (name) => {
    const session = get().savedSessions.find((ss) => ss.name === name);
    if (!session) return;
    try {
      const restored = studioSavedSessionRestoreState(session.state);
      set({
        ...restored,
        ...compilerConfigurationInvalidatedState(),
      });
      runRestoredExperiment();
    } catch (error) {
      set(studioProjectFailureState(error, "Failed to restore the saved session"));
    }
  },

  deleteSession: (name) => {
    const nextState = studioSavedSessionRemovedState(get().savedSessions, name);
    const result = writeStoredStudioSessions(nextState.savedSessions);
    set({ ...nextState, sessionWriteResult: result });
    return result;
  },

  applyShareLink: async () => {
    // The Share button has always produced a link. Nothing ever read one back:
    // `readStudioStartupHashState` had no caller outside its own test, so a
    // colleague opening the link got the default Studio and no indication that
    // the link had carried anything. A link handed out that does nothing is
    // worse than no link, because the button promises otherwise.
    //
    // The judgement lives in `studioShareLinkDecision`; this only carries it out.
    const knownModels = get().models.map((model) => model.name);
    // A model link opens one model and leaves the run settings alone.
    const modelLink = studioModelLinkDecision(readStudioModelLink(), knownModels);
    if (modelLink.kind === "unknown-model") {
      set({ linkNotice: modelLink.message });
      return;
    }
    if (modelLink.kind === "select") {
      await get().selectModel(modelLink.modelName);
      return;
    }
    const decision = studioShareLinkDecision(readStudioStartupHashState(), knownModels);
    if (decision.kind === "none") return;
    if (decision.kind === "unknown-model") {
      set({ linkNotice: decision.message });
      return;
    }
    set({
      current: decision.current,
      duration: decision.duration,
      protocol: decision.protocol,
    });
    await get().selectModel(decision.modelName);
  },

  shareURL: () => {
    const s = get();
    void copyStudioShareUrlInRuntime({
      sourceMode: s.sourceMode,
      selectedModelName: s.selectedModelName,
      equations: s.equations,
      threshold: s.threshold,
      reset: s.reset,
      modelParams: s.modelParams,
      odeParams: s.odeParams,
      odeInit: s.odeInit,
      dt: s.dt,
      duration: s.duration,
      current: s.current,
      protocol: s.protocol,
    }).then((result) => {
      set(studioShareStatusState(result));
      scheduleStudioShareStatusClear(() => { set(studioShareStatusClearedState()); });
    });
  },
  };
}
