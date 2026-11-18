// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio experiment workspace actions

import { fetchTemplates, fetchModels, fetchModelDetail, fetchPresets, fetchPreset } from "../api/client";
import { studioProjectFailureState } from "../studioProjectState";
import { compilerConfigurationInvalidatedState, compilerCosimInvalidatedState } from "../compilerStoreState";
import {
  modelDetailLoadedState,
  modelSelectionStartedState,
  modelsLoadedState,
  presetSelection,
  presetsLoadedState,
  templateSelectedState,
  templatesLoadedState,
} from "../modelSelectionStoreState";
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
  numberRecordEntryState,
  protocolState,
  resetState,
  sourceModeState,
  sweepParamState,
  sweepParamYState,
  thresholdState,
} from "../studioInputState";
import type { StudioState, StudioStoreActions } from "./studioTypes";

/**
 * The experiment workspace's actions: the inputs being edited, the template, model and preset catalogues they are
 * chosen from, the choices themselves, and the return to a model's defaults.
 */
export type StudioWorkspaceActions = Pick<StudioStoreActions,
  "setSourceMode" | "setEquations" | "setThreshold" | "setReset" | "setOdeParam" | "setOdeInit" |
  "setModelParam" | "setModelIntegrator" | "setModelQFormat" | "setDt" | "setDuration" | "setCurrent" |
  "setProtocol" | "setFrequencyHz" | "setSeed" | "setTrial" | "setActiveTab" | "setCandidateDraft" |
  "setModelFilter" | "setSweepParam" | "setSweepParamY" | "loadTemplates" | "loadModels" | "loadPresets" |
  "selectTemplate" | "selectModel" | "loadPreset" | "resetDefaults">;

/** The workspace actions, and the step that shows a restored experiment. */
export interface StudioWorkspaceActionBinding {
  /** The actions the store exposes. */
  readonly actions: StudioWorkspaceActions;
  /**
   * Show a restored experiment: load the restored model's details without replacing the restored inputs, then run
   * the simulation. The project and saved-session actions call it after they restore a workspace.
   */
  readonly runRestoredExperiment: () => void;
}

/**
 * Bind the experiment workspace to the Studio store.
 *
 * The inputs of the experiment (the source, the equations and parameters, the stimulus and what is on screen) are
 * edited here, and the catalogues of templates, models and presets are loaded and chosen from. A choice of model, a
 * switch of source and a restored experiment each outdate the model details still on their way; the counters that do
 * so live in this closure, which is why the restore step is returned with the actions.
 *
 * @param set - Writes a patch into the store.
 * @param get - Reads the store as it stands.
 * @returns The actions and the restore step.
 */
export function createStudioWorkspaceActions(
  set: (partial: Partial<StudioState> | ((state: StudioState) => Partial<StudioState>)) => void,
  get: () => StudioState,
): StudioWorkspaceActionBinding {
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

  const actions: StudioWorkspaceActions = {
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

  resetDefaults: () => {
    const s = get();
    if (s.sourceMode === "model" && s.modelDetail) {
      set({ ...modelDefaultsState(s.modelDetail), ...compilerConfigurationInvalidatedState() });
    }
    void get().runSimulation();
  },
  };

  return { actions, runRestoredExperiment };
}
