// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio Zustand store
// Studio store action implementations (mutations + async side effects).

import {
  graphEditRecorded,
  graphRedone,
  graphSnapshotOf,
  graphUndone,
  isLayoutOnlyUpdate,
  type StudioGraphHistory,
  type StudioGraphHistoryStep,
  type StudioGraphSnapshot,
} from "../studioGraphHistory";
import type { PopulationNode, ProjectionEdge } from "../api/client";
import { createStoreArtifactDownloader } from "./studioArtifactDownload";
import { readStudioStartupHashState } from "../studioStartupRuntime";
import { studioShareLinkDecision } from "../shareLinkApplication";
import {
  fetchTemplates,
  fetchModels,
  fetchModelDetail,
  fetchPresets,
  fetchPreset,
  fetchStudioAuthSession,
  createStudioAuditQuarantineArchive,
  fetchStudioIdentityServiceAccounts,
  fetchStudioAuditQuarantineArchiveRetention,
  fetchStudioAuditExport,
  fetchStudioAuditStatus,
  fetchStudioCapabilities,
  fetchStudioJobArtifact,
  fetchStudioJobs,
  fetchStudioJobStatus,
  fetchStudioOperatorStatus,
  fetchStudioIdentityBrowserUsers,
  loginStudioBrowserUser,
  logoutStudioBrowserUser,
  purgeStudioAuditQuarantineArchiveRetention,
  restoreStudioAuditQuarantineArchive,
  rotateStudioIdentityBrowserUserPassword,
  simulateODE,
  simulateModel,
  fetchPrecision,
  fetchCodegen,
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
  fetchSurrogates as apiFetchSurrogates,
  startTraining as apiStartTraining,
  stopTraining as apiStopTraining,
  exportTrainingCheckpoint as apiExportTrainingCheckpoint,
  importTrainingCheckpoint as apiImportTrainingCheckpoint,
  restoreTrainingWeights as apiRestoreTrainingWeights,
  attachTrainingWeights as apiAttachTrainingWeights,
  attachTrainingWeightsLive as apiAttachTrainingWeightsLive,
  fetchGraphModels as apiFetchGraphModels,
  createPopulation as apiCreatePop,
  createProjection as apiCreateProj,
  simulateGraph as apiSimGraph,
  graphModelContract as apiGraphModelContract,
  validateGraph as apiValidateGraph,
  exportNIR as apiExportNIR,
  importNIR as apiImportNIR,
  branchRefusedEdit as apiBranchRefusedEdit,
  saveProject as apiSaveProject,
  loadProject as apiLoadProject,
  listDeletedProjects as apiListDeletedProjects,
  restoreProject as apiRestoreProject,
  listProjects as apiListProjects,
  deleteProject as apiDeleteProject,
  runPipeline as apiRunPipeline,
  createStudioIdentityBrowserUser,
  setStudioAuthToken,
  updateStudioIdentityBrowserUser,
  updateStudioIdentityServiceAccount,
  validateStudioAuditQuarantineArchive,
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
  auditArchiveCreatedState,
  auditArchivePurgedState,
  auditArchiveRestoredState,
  auditArchiveRetentionLoadedState,
  auditArchiveValidationLoadedState,
  auditExportLoadedState,
  auditFailureState,
  auditLoadingState,
  auditStatusLoadedState,
} from "../auditShell";
import {
  studioSavedSessionRemovedState,
  studioSavedSessionRestoreState,
  studioSavedSessionState,
  studioSavedSessionUpsertState,
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
  studioDefaultPopulationRequest,
  studioDefaultProjectionRequest,
  studioGraphFailureState,
  studioGraphImportedState,
  studioGraphModelsLoadedState,
  studioGraphRequest,
  studioGraphSimulationCompletedState,
  studioGraphSimulationStartState,
  studioGraphWithoutPopulation,
  studioPipelineCompletedState,
  studioPipelineStartState,
  studioPopulationAddedState,
  studioPopulationUpdatedState,
  studioProjectionAddedState,
  studioProjectionRemovedState,
  studioProjectionUpdatedState,
} from "../studioGraphRequests";
import {
  studioGraphValidatedState,
  studioGraphValidationLocatedState,
} from "../studioGraphValidation";
import { studioDuplicatePlan, studioDuplicateSummary } from "../studioGraphDuplicate";
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
  studioAnalysisIdleState,
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
  studioSimulationResultState,
  studioSTAResultState,
} from "../studioAnalysisState";
import {
  replayPackExport,
  simulationExportPlan,
} from "../simulationExports";
import { downloadBrowserArtefact } from "../browserArtefactDownload";
import {
  networkNirExportPlan,
} from "../networkNirExport";
import {
  parseTrainingCheckpointPayload,
} from "../trainingCheckpoint";
import {
  verifyTrainingWeightArtifactBlob,
} from "../trainingRestore";
import {
  connectStudioTrainingEventSource,
} from "../studioTrainingStream";
import {
  trainingCheckpointExportPlan,
  trainingWeightRestoreVerificationExportPlan,
} from "../trainingExports";
import {
  trainingCheckpointImportedState,
  trainingConfigUpdatedState,
  trainingEpochAppendedState,
  trainingExportSuccessState,
  trainingFailureState,
  trainingPreconditionErrorState,
  trainingStartedState,
  trainingStartState,
  trainingStoppingState,
  trainingStreamDisconnectedState,
  trainingStreamErrorState,
  trainingSurrogatesLoadedState,
  trainingTerminalState,
  trainingWeightRestoreVerificationLoadedState,
  trainingWeightMaterializationLoadedState,
  trainingWeightAttachLoadedState,
  trainingWeightLiveAttachLoadedState,
  trainingWeightRestoreVerificationStartState,
} from "../trainingStoreState";
import {
} from "../evidenceBundles";
import {
  adminBusyState,
  adminFailureState,
  identityAccountsLoadedState,
  identityAccountsMutatedState,
  jobStatusLoadedState,
  operatorStatusLoadedState,
} from "../adminStoreState";
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
import { studioExperimentKey, studioPrecisionKey, studioTrainingKey } from "../studioExperimentKey";
import { runStoreCompile } from "./studioCompile";
import { runStoreSynthesis } from "./studioSynthesis";
import { runStoreBundle } from "./studioBundle";

/**
 * The graph the store currently holds, as a history snapshot.
 *
 * @param state - The store's graph fields.
 * @returns The snapshot.
 */
function studioGraphSnapshotFrom(state: {
  graphPopulations: PopulationNode[];
  graphProjections: ProjectionEdge[];
}): StudioGraphSnapshot {
  return graphSnapshotOf(state.graphPopulations, state.graphProjections);
}

/**
 * The history after recording the graph as it stands before an edit.
 *
 * The snapshot is taken *before* the edit, which is what makes undo restore
 * the state the reader was looking at rather than the one they just created.
 *
 * @param state - The store's graph and history fields.
 * @returns The history with that snapshot recorded.
 */
function studioGraphEditRecordedFrom(state: {
  graphHistory: StudioGraphHistory;
  graphPopulations: PopulationNode[];
  graphProjections: ProjectionEdge[];
}): StudioGraphHistory {
  return graphEditRecorded(state.graphHistory, studioGraphSnapshotFrom(state));
}

/**
 * The state patch that reinstates one recorded graph.
 *
 * @param step - The history step to reinstate.
 * @returns The patch: the graph as it was, and the history that produced it.
 */
function studioGraphRestoredState(step: StudioGraphHistoryStep): {
  graphHistory: StudioGraphHistory;
  graphPopulations: PopulationNode[];
  graphProjections: ProjectionEdge[];
} {
  return {
    graphHistory: step.history,
    graphPopulations: step.snapshot.populations,
    graphProjections: step.snapshot.projections,
  };
}


/**
 * The pending auto-simulation, if one is scheduled.
 *
 * Module-level rather than in the store because it is a timer handle, not
 * state: nothing renders from it, and putting it in the store would make every
 * keystroke that reschedules a run also re-render every subscriber.
 */
let debounceTimer: StudioAutoSimulationTimer | null = null;

/**
 * Which of the store's experiment keys a result belongs to.
 *
 * The workflow asks three separate questions -- has the current experiment been
 * simulated, analysed, trained -- so results record against three keys rather
 * than one.
 */
type StudioExperimentKeyField =
  | "analysisExperimentKey"
  | "resultExperimentKey"
  | "trainingExperimentKey";

/**
 * Apply a result only while the experiment that produced it is still on screen.
 *
 * A run started before the reader changed the model can still be in flight
 * when they change it. Landing its response on top of the new configuration
 * would put another experiment's trace in front of them and let the guided
 * workflow report a completed simulation of a run nobody asked for.
 *
 * A superseded response is therefore dropped rather than applied -- but the
 * busy flag is cleared regardless, because the run it belonged to really has
 * ended and leaving the panel spinning would be a second lie on top of the
 * first.
 *
 * @param context - The store accessors, the key captured when the request was
 *   submitted, the field to record it in, and the patch to apply.
 * @returns Whether the result was applied.
 */
function applyResultForExperiment(context: {
  get: () => StudioState;
  set: (partial: Partial<StudioState>) => void;
  requestedKey: string;
  field: StudioExperimentKeyField;
  patch: Partial<StudioState>;
}): boolean {
  const currentKey = studioExperimentKey(simulationConfigInput(context.get()));
  if (currentKey !== context.requestedKey) {
    context.set(studioAnalysisIdleState());
    return false;
  }
  context.set({ ...context.patch, [context.field]: context.requestedKey });
  return true;
}

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
 * @returns Every action, complete. The return type is the derived action half
 *   of the state rather than a `Partial`, so a forgotten action is a type
 *   error here instead of a runtime `undefined` at the first click.
 */
export function createStudioStoreActions(
  set: (partial: Partial<StudioState> | ((state: StudioState) => Partial<StudioState>)) => void,
  get: () => StudioState,
): StudioStoreActions {
  return {
  setSourceMode: (m) => { set({ ...sourceModeState(m), ...compilerConfigurationInvalidatedState() }); },
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
  setSeed: (seed) => { set(seedState(seed)); get().autoSimulate(); },
  setTrial: (trial) => { set(trialState(trial)); get().autoSimulate(); },
  setActiveTab: (tab) => { set(activeTabState(tab)); },
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
    set(studioAuthLoadingState());
    try {
      await logoutStudioBrowserUser();
    } catch (error: unknown) {
      set(studioAuthLogoutFailureState(error));
    } finally {
      clearStoredStudioAuthToken();
      syncStoredStudioAuthToken(setStudioAuthToken);
      set(studioAuthLogoutCompleteState());
    }
  },
  loadAuditStatus: async () => {
    set(auditLoadingState());
    try {
      const auditStatus = await fetchStudioAuditStatus();
      set(auditStatusLoadedState(auditStatus));
    } catch (error: unknown) {
      set(auditFailureState(error, "Audit status check failed"));
    }
  },
  loadAuditExport: async () => {
    set(auditLoadingState());
    try {
      const auditExport = await fetchStudioAuditExport(100);
      set(auditExportLoadedState(auditExport));
    } catch (error: unknown) {
      set(auditFailureState(error, "Audit export failed"));
    }
  },
  createAuditQuarantineArchive: async (limit) => {
    set(auditLoadingState());
    try {
      const auditArchive = await createStudioAuditQuarantineArchive(limit);
      const [operatorStatus, jobList, auditExport] = await Promise.all([
        fetchStudioOperatorStatus(),
        fetchStudioJobs(),
        fetchStudioAuditExport(100),
      ]);
      set(auditArchiveCreatedState(auditArchive, auditExport, operatorStatus, jobList));
    } catch (error: unknown) {
      set(auditFailureState(error, "Audit archive creation failed"));
    }
  },
  loadAuditQuarantineArchiveRetention: async (retainLatest) => {
    set(auditLoadingState());
    try {
      const auditArchiveRetention = await fetchStudioAuditQuarantineArchiveRetention(retainLatest);
      set(auditArchiveRetentionLoadedState(auditArchiveRetention));
    } catch (error: unknown) {
      set(auditFailureState(error, "Audit archive retention check failed"));
    }
  },
  validateAuditQuarantineArchive: async (archive, manifest) => {
    set(auditLoadingState());
    try {
      const auditArchiveValidation = await validateStudioAuditQuarantineArchive(archive, manifest);
      set(auditArchiveValidationLoadedState(auditArchiveValidation));
    } catch (error: unknown) {
      set(auditFailureState(error, "Audit archive validation failed"));
    }
  },
  restoreAuditQuarantineArchive: async (archive, manifest) => {
    set(auditLoadingState());
    try {
      const auditArchiveRestore = await restoreStudioAuditQuarantineArchive(archive, manifest);
      const [operatorStatus, jobList] = await Promise.all([
        fetchStudioOperatorStatus(),
        fetchStudioJobs(),
      ]);
      set(auditArchiveRestoredState(auditArchiveRestore, operatorStatus, jobList));
    } catch (error: unknown) {
      set(auditFailureState(error, "Audit archive restore failed"));
    }
  },
  purgeAuditQuarantineArchiveRetention: async (retainLatest) => {
    set(auditLoadingState());
    try {
      const auditArchivePurge = await purgeStudioAuditQuarantineArchiveRetention(retainLatest);
      const [operatorStatus, jobList, auditArchiveRetention] = await Promise.all([
        fetchStudioOperatorStatus(),
        fetchStudioJobs(),
        fetchStudioAuditQuarantineArchiveRetention(retainLatest),
      ]);
      set(auditArchivePurgedState(
        auditArchivePurge,
        auditArchiveRetention,
        operatorStatus,
        jobList,
      ));
    } catch (error: unknown) {
      set(auditFailureState(error, "Audit archive retention purge failed"));
    }
  },
  createEvidenceBundle: (request) => runStoreBundle("admin", request, get, set),
  createEvidenceBundleForSurface: (surface, request) => runStoreBundle(surface, request, get, set),
  downloadEvidenceBundleArtifact: async (relativePath) => {
    await get().downloadEvidenceBundleArtifactForSurface("admin", relativePath);
  },
  downloadEvidenceBundleArtifactForSurface: createStoreArtifactDownloader(get, set),
  loadJobStatus: async () => {
    set(adminBusyState());
    try {
      const [jobStatus, jobList] = await Promise.all([
        fetchStudioJobStatus(),
        fetchStudioJobs(),
      ]);
      set(jobStatusLoadedState(jobStatus, jobList));
    } catch (error: unknown) {
      set(adminFailureState(error, "Job status check failed"));
    }
  },
  loadIdentityServiceAccounts: async () => {
    set(adminBusyState());
    try {
      const [accountsResponse, usersResponse] = await Promise.all([
        fetchStudioIdentityServiceAccounts(),
        fetchStudioIdentityBrowserUsers(),
      ]);
      set(identityAccountsLoadedState(accountsResponse, usersResponse));
    } catch (error: unknown) {
      set(adminFailureState(error, "Identity account check failed"));
    }
  },
  createIdentityBrowserUser: async (create) => {
    set(adminBusyState());
    try {
      await createStudioIdentityBrowserUser(create);
      const [accountsResponse, usersResponse, auditExport] = await Promise.all([
        fetchStudioIdentityServiceAccounts(),
        fetchStudioIdentityBrowserUsers(),
        fetchStudioAuditExport(100),
      ]);
      set(identityAccountsMutatedState(accountsResponse, usersResponse, auditExport));
    } catch (error: unknown) {
      set(adminFailureState(error, "Browser user creation failed"));
    }
  },
  updateIdentityServiceAccount: async (principalId, update) => {
    set(adminBusyState());
    try {
      await updateStudioIdentityServiceAccount(principalId, update);
      const [accountsResponse, usersResponse, auditExport] = await Promise.all([
        fetchStudioIdentityServiceAccounts(),
        fetchStudioIdentityBrowserUsers(),
        fetchStudioAuditExport(100),
      ]);
      set(identityAccountsMutatedState(accountsResponse, usersResponse, auditExport));
    } catch (error: unknown) {
      set(adminFailureState(error, "Identity account update failed"));
    }
  },
  updateIdentityBrowserUser: async (username, update) => {
    set(adminBusyState());
    try {
      await updateStudioIdentityBrowserUser(username, update);
      const [accountsResponse, usersResponse, auditExport] = await Promise.all([
        fetchStudioIdentityServiceAccounts(),
        fetchStudioIdentityBrowserUsers(),
        fetchStudioAuditExport(100),
      ]);
      set(identityAccountsMutatedState(accountsResponse, usersResponse, auditExport));
    } catch (error: unknown) {
      set(adminFailureState(error, "Browser user update failed"));
    }
  },
  rotateIdentityBrowserUserPassword: async (username, password) => {
    set(adminBusyState());
    try {
      await rotateStudioIdentityBrowserUserPassword(username, { password });
      const [accountsResponse, usersResponse, auditExport] = await Promise.all([
        fetchStudioIdentityServiceAccounts(),
        fetchStudioIdentityBrowserUsers(),
        fetchStudioAuditExport(100),
      ]);
      set(identityAccountsMutatedState(accountsResponse, usersResponse, auditExport));
    } catch (error: unknown) {
      set(adminFailureState(error, "Browser user secret rotation failed"));
    }
  },
  loadOperatorStatus: async () => {
    set(adminBusyState());
    try {
      const [operatorStatus, jobList] = await Promise.all([
        fetchStudioOperatorStatus(),
        fetchStudioJobs(),
      ]);
      set(operatorStatusLoadedState(operatorStatus, jobList));
    } catch (error: unknown) {
      set(adminFailureState(error, "Operator status check failed"));
    }
  },
  loadModels: async () => {
    const models = await fetchModels();
    set(modelsLoadedState(models));
    const firstModel = models[0];
    if (firstModel !== undefined && !get().selectedModelName) await get().selectModel(firstModel.name);
  },
  loadPresets: async () => { set(presetsLoadedState(await fetchPresets())); },

  selectTemplate: (name) => {
    const template = get().templates.find((candidate) => candidate.name === name);
    if (template === undefined) return;
    set({ ...templateSelectedState(template), ...compilerConfigurationInvalidatedState() });
    void get().runSimulation();
  },

  selectModel: async (name) => {
    set({ ...modelSelectionStartedState(name), ...compilerConfigurationInvalidatedState() });
    // No `detail === null` guard: the route's contract does not allow one, and
    // the guard that used to be here defended a single field of a response that
    // nothing validates. Response validation at the API boundary is recorded as
    // its own unit.
    const detail = await fetchModelDetail(name);
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

  runSimulation: async () => {
    const s = get();
    if (s.isSimulating) return;
    const requestedKey = studioExperimentKey(simulationConfigInput(s));
    // Keep the previous trace visible, but only the new successful response may
    // restore completion evidence. A pending or failed rerun is not a success.
    set({ ...studioAnalysisStartState(), resultExperimentKey: null });
    try {
      const cfg = studioSimulationConfig(simulationConfigInput(s));
      const result = s.sourceMode === "model" && s.selectedModelName
        ? await simulateModel(cfg) : await simulateODE(cfg);
      applyResultForExperiment({
        field: "resultExperimentKey",
        get,
        patch: studioSimulationResultState(result),
        requestedKey,
        set,
      });
    } catch (e) {
      set(studioAnalysisFailureState(e));
    }
  },

  runFICurve: () => runStoreHeavyAnalysis("fi_curve", get, set),
  runBifurcation: () => runStoreHeavyAnalysis("bifurcation", get, set),
  runSensitivity: () => runStoreHeavyAnalysis("sensitivity", get, set),

  runPrecision: () => runStoreDirectAnalysis(get, set, async (s) => {
    if (s.sourceMode !== "ode") {
      throw new Error("Precision compare only for custom ODE mode");
    }
    const precResult = await fetchPrecision(
      studioPrecisionRequest(simulationConfigInput(s), s.modelQFormat),
    );
    return studioPrecisionResultState(precResult);
  }, "precision", (s) => studioPrecisionKey(simulationConfigInput(s), s.modelQFormat)),

  runCompile: () => runStoreCompile("compile", get, set),

  runCosim: () => runStoreCompile("cosim", get, set),

  runHeatmap: () => runStoreHeavyAnalysis("heatmap", get, set),

  runCodegen: async () => {
    const s = get();
    set(studioCodegenStartState());
    try {
      const res = await fetchCodegen(studioExperimentExportRequest(simulationConfigInput(s)));
      set(studioCodegenResultState(res.script, res.oneliner, res.replay_script, res.experiment_sha256));
    } catch (e) { set(studioAnalysisErrorState(e instanceof Error ? e.message : String(e))); }
  },

  exportReplayPack: async () => {
    const s = get();
    try {
      const pack = await fetchReplayPack(studioExperimentExportRequest(simulationConfigInput(s)));
      const artefact = replayPackExport(pack);
      downloadBrowserArtefact(artefact.blob, artefact.filename);
    } catch (e) { set(studioAnalysisErrorState(e instanceof Error ? e.message : String(e))); }
  },

  exportData: () => {
    const plan = simulationExportPlan("json", get().result);
    if (plan.available) {
      plan.writeArtefact();
    }
  },

  exportCSV: () => {
    const plan = simulationExportPlan("csv", get().result);
    if (plan.available) {
      plan.writeArtefact();
    }
  },

  exportSVG: () => {
    const plan = simulationExportPlan("svg", get().result);
    if (plan.available) {
      plan.writeArtefact();
    } else {
      plan.runFallback();
    }
  },

  runCharacterize: () => {
    runStoreCharacterize(get, set);
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

  runCompare: (configB) => runStoreDirectAnalysis(get, set, async (s) => {
    const configA = studioSimulationConfig(simulationConfigInput(s));
    const compareResult = await fetchCompare(configA, configB);
    return studioCompareResultState(compareResult);
  }, "compare"),

  runNullclines: () => runStoreDirectAnalysis(get, set, async (s) => {
    if (s.sourceMode !== "ode" || s.equations.length < 2) {
      throw new Error("Nullclines need 2+ variable ODE in custom mode");
    }
    const vars = Object.keys(s.odeInit);
    const [var0, var1] = vars;
    if (var0 === undefined || var1 === undefined) throw new Error("Nullclines need initial values for two variables");
    const v0vals = s.result?.states[var0];
    const v1vals = s.result?.states[var1];
    const r0: [number, number] = v0vals
      ? [Math.min(...v0vals) - 10, Math.max(...v0vals) + 10]
      : [-80, 40];
    const r1: [number, number] = v1vals
      ? [Math.min(...v1vals) - 0.5, Math.max(...v1vals) + 0.5]
      : [-2, 2];
    const nullclineResult = await fetchNullclines(
      studioNullclineRequest({
        equations: s.equations,
        odeParams: s.odeParams,
        odeInit: s.odeInit,
        protocol: s.protocol,
        current: s.current,
        ranges: { [var0]: r0, [var1]: r1 },
        gridSize: 60,
      }),
    );
    return studioNullclineResultState(nullclineResult);
  }),

  runFreqResponse: () => runStoreDirectAnalysis(get, set, async (s) => {
    const cfg = studioSimulationConfig(simulationConfigInput(s));
    const freqResult = await fetchFreqResponse(studioFrequencyResponseRequest(cfg, s.current));
    return studioFrequencyResultState(freqResult);
  }, "freq"),

  computeSTA: () => {
    const { result } = get();
    if (!result) return;
    const state = studioSTAResultState(result);
    if (state !== null) set(state);
  },

  runBuildIR: () => runStoreCompile("ir", get, set),

  runEmitSV: () => runStoreCompile("sv", get, set),

  setSynthTarget: (t) => { set(synthesisTargetState(t)); },

  runSynthesis: () => runStoreSynthesis(get, set, async (s) => {
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

  runMultiTargetSynthesis: () => runStoreSynthesis(get, set, async (s) => {
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
    const state = studioProjectSaveState(get());
    const expected = studioProjectExpectedRevision(get().projectRevision, name);
    try {
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
      void get().runSimulation();
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
    } catch (e) { set(studioGraphFailureState(e, "Pipeline run failed", { clearBusy: true })); }
  },

  loadGraphModels: async () => {
    try {
      const graphModels = await apiFetchGraphModels();
      set(studioGraphModelsLoadedState(graphModels));
    } catch { /* non-critical */ }
  },

  addPopulation: async (neuronType) => {
    const s = get();
    try {
      const pop = await apiCreatePop(studioDefaultPopulationRequest(neuronType, s.graphPopulations.length));
      set((prev) => ({
        ...studioPopulationAddedState(prev.graphPopulations, pop),
        graphHistory: studioGraphEditRecordedFrom(prev),
      }));
    } catch (e) { set(studioGraphFailureState(e, "Population creation failed")); }
  },

  removePopulation: (id) => {
    set((s) => {
      const graph = studioGraphWithoutPopulation({
        populations: s.graphPopulations,
        projections: s.graphProjections,
      }, id);
      if (graph.populations.length === s.graphPopulations.length) {
        return {};
      }
      const survives = graph.projections.some(
        (projection) => projection.id === s.selectedProjectionId,
      );
      return {
        graphHistory: studioGraphEditRecordedFrom(s),
        graphPopulations: graph.populations,
        graphProjections: graph.projections,
        // Deleting a population takes its projections with it, and one of
        // them may be the projection the editor is editing.
        selectedPopulationId: s.selectedPopulationId === id ? null : s.selectedPopulationId,
        selectedProjectionId: survives ? s.selectedProjectionId : null,
      };
    });
  },

  updatePopulation: (id, updates) => {
    set((s) => {
      const patch = studioPopulationUpdatedState(s.graphPopulations, id, updates);
      // A drag writes a position on every frame; those are not edits to undo.
      if (isLayoutOnlyUpdate(updates)) {
        return patch;
      }
      return { ...patch, graphHistory: studioGraphEditRecordedFrom(s) };
    });
  },

  undoGraphEdit: () => {
    set((s) => {
      const step = graphUndone(s.graphHistory, studioGraphSnapshotFrom(s));
      return step === null ? {} : studioGraphRestoredState(step);
    });
  },

  redoGraphEdit: () => {
    set((s) => {
      const step = graphRedone(s.graphHistory, studioGraphSnapshotFrom(s));
      return step === null ? {} : studioGraphRestoredState(step);
    });
  },

  addProjection: async (sourceId, targetId) => {
    const source = get().graphPopulations.find((population) => population.id === sourceId);
    if (!source) {
      set(studioGraphFailureState(new Error(`Source population ${sourceId} not found`), "Projection creation failed"));
      return;
    }
    try {
      const proj = await apiCreateProj(studioDefaultProjectionRequest(sourceId, targetId, source.neuron_type));
      set((prev) => ({
        ...studioProjectionAddedState(prev.graphProjections, proj),
        graphHistory: studioGraphEditRecordedFrom(prev),
      }));
    } catch (e) { set(studioGraphFailureState(e, "Projection creation failed")); }
  },

  removeProjection: (id) => {
    set((s) => {
      const patch = studioProjectionRemovedState(s.graphProjections, id);
      if (patch.graphProjections.length === s.graphProjections.length) {
        return {};
      }
      return {
        ...patch,
        graphHistory: studioGraphEditRecordedFrom(s),
        // Editing a projection that is no longer in the graph would write
        // fields into nothing.
        selectedProjectionId: s.selectedProjectionId === id ? null : s.selectedProjectionId,
      };
    });
  },

  updateProjection: (id, updates) => {
    set((s) => ({
      ...studioProjectionUpdatedState(s.graphProjections, id, updates),
      graphHistory: studioGraphEditRecordedFrom(s),
    }));
  },

  selectProjection: (id) => {
    set({ selectedProjectionId: id, selectedPopulationId: null });
  },

  selectPopulations: (ids) => {
    set({ selectedPopulationIds: ids });
  },

  duplicateSelection: async () => {
    const s = get();
    const plan = studioDuplicatePlan(s.graphPopulations, s.graphProjections, s.selectedPopulationIds);
    if (plan.populations.length === 0) {
      set({ graphNotice: null });
      return;
    }
    const before = studioGraphEditRecordedFrom(s);
    try {
      // Identity is the server's: a client that minted ids would be a second
      // implementation of it, free to collide with the first.
      const created = new Map<string, string>();
      const populations: PopulationNode[] = [];
      for (const copy of plan.populations) {
        const made = await apiCreatePop({
          count: copy.count,
          drive: copy.drive,
          label: copy.label,
          model: copy.model,
          neuron_type: copy.neuron_type,
          params: copy.params,
          x: copy.x,
          y: copy.y,
        });
        created.set(copy.sourceId, made.id);
        populations.push(made);
      }
      const projections: ProjectionEdge[] = [];
      for (const copy of plan.projections) {
        const source = created.get(copy.sourceId);
        const target = created.get(copy.targetId);
        if (source === undefined || target === undefined) {
          // The plan only ever carries projections whose both endpoints were
          // copied, so this cannot happen — and if it ever did, dropping the
          // projection would hand back a graph quietly smaller than the one
          // the user asked for, which reads as success.
          throw new Error(
            `Duplicate produced a projection whose endpoint was not copied: ${copy.sourceId} → ${copy.targetId}`,
          );
        }
        const made = await apiCreateProj({
          delay: copy.delay,
          probability: copy.probability,
          rule: copy.rule,
          source_id: source,
          target_id: target,
          weight: copy.weight,
        });
        // The creation route carries neither seed nor autapses, and both are
        // executed; they are applied here so the copy runs as its original does.
        projections.push({
          ...made,
          ...(copy.autapses === undefined ? {} : { autapses: copy.autapses }),
          ...(copy.seed === undefined ? {} : { seed: copy.seed }),
        });
      }
      set((prev) => ({
        graphHistory: before,
        graphNotice: studioDuplicateSummary(plan),
        graphPopulations: [...prev.graphPopulations, ...populations],
        graphProjections: [...prev.graphProjections, ...projections],
        selectedPopulationIds: populations.map((population) => population.id),
      }));
    } catch (e) {
      set(studioGraphFailureState(e, "Duplicating the selection failed"));
    }
  },

  selectPopulation: (id) => {
    set({ selectedPopulationId: id, selectedProjectionId: null });
    if (id === null) return;
    const population = get().graphPopulations.find((one) => one.id === id);
    if (population !== undefined) {
      void get().loadPopulationModelContract(population.model);
    }
  },

  loadPopulationModelContract: async (model) => {
    if (get().populationModelContract?.model === model) return;
    try {
      set({ populationModelContract: await apiGraphModelContract(model) });
    } catch (e) {
      // The parameters simply are not offered without their contract; the
      // identity and drive fields still are, and the failure is reported
      // rather than shown as a model with no parameters.
      set({
        ...studioGraphFailureState(e, `Model contract for ${model} could not be loaded`),
        populationModelContract: null,
      });
    }
  },

  validateGraphAction: async () => {
    const s = get();
    try {
      const graph = studioGraphRequest(
        s.graphPopulations,
        s.graphProjections,
        s.duration,
        s.dt,
        s.seed,
      );
      const validation = await apiValidateGraph(graph);
      set(studioGraphValidatedState(validation, s.graphPopulations, s.graphProjections));
    } catch (e) {
      set(studioGraphFailureState(e, "Graph validation failed"));
    }
  },

  simulateGraphAction: async () => {
    const s = get();
    if (s.isSimulating) return;
    set(studioGraphSimulationStartState());
    try {
      const graph = studioGraphRequest(s.graphPopulations, s.graphProjections, s.duration, s.dt, s.seed);
      const validation = await apiValidateGraph(graph);
      if (!validation.valid) {
        set(
          studioGraphValidationLocatedState(validation, s.graphPopulations, s.graphProjections),
        );
        return;
      }
      const graphSimResult = await apiSimGraph(graph);
      set(studioGraphSimulationCompletedState(graphSimResult));
    } catch (e) { set(studioGraphFailureState(e, "Graph simulation failed", { clearBusy: true })); }
  },

  exportGraphNIR: async () => {
    const s = get();
    try {
      const nir = await apiExportNIR({ populations: s.graphPopulations, projections: s.graphProjections });
      networkNirExportPlan(nir).writeArtefact();
    } catch (e) { set(studioGraphFailureState(e, "Graph NIR export failed")); }
  },

  importGraphNIR: async (nir) => {
    try {
      const graph = await apiImportNIR(nir);
      set(studioGraphImportedState(graph));
    } catch (e) { set(studioGraphFailureState(e, "Graph NIR import failed")); }
  },

  loadSurrogates: async () => {
    try {
      const trainingSurrogates = await apiFetchSurrogates();
      set(trainingSurrogatesLoadedState(trainingSurrogates));
    } catch { /* non-critical */ }
  },

  startTraining: async () => {
    const s = get();
    if (s.trainingStatus === "running") return;
    // Recorded at the start rather than at the end: what makes a finished run
    // stale is a change to what was trained, and the reader can change that
    // while the run is going.
    set({ ...trainingStartState(), trainingExperimentKey: studioTrainingKey(s.trainingConfig) });
    try {
      const result = await apiStartTraining(s.trainingConfig);
      set(trainingStartedState(result.job_id));
      connectStudioTrainingEventSource(result.job_id, {
        onDisconnected: () => { set(trainingStreamDisconnectedState()); },
        onEpoch: (metrics) => { set((prev) => trainingEpochAppendedState(prev.trainingEpochs, metrics)); },
        onError: (message) => { set(trainingStreamErrorState(message)); },
        onTerminal: (status) => { set(trainingTerminalState(status)); },
      });
    } catch (e) {
      set(trainingFailureState(e, "Training start failed", { markFailed: true }));
    }
  },

  stopTraining: async () => {
    const s = get();
    if (!s.trainingJobId) return;
    try {
      await apiStopTraining(s.trainingJobId);
      set(trainingStoppingState());
    } catch (e) {
      set(trainingFailureState(e, "Training stop failed"));
    }
  },

  exportTrainingCheckpoint: async () => {
    const s = get();
    if (!s.trainingJobId) return;
    try {
      const checkpoint = await apiExportTrainingCheckpoint(s.trainingJobId);
      trainingCheckpointExportPlan(checkpoint).writeExport();
    } catch (e) {
      set(trainingFailureState(e, "Training checkpoint export failed"));
    }
  },

  importTrainingCheckpointText: async (checkpointJson) => {
    try {
      const parsed = parseTrainingCheckpointPayload(checkpointJson);
      const imported = await apiImportTrainingCheckpoint(parsed);
      set((s) => trainingCheckpointImportedState(s.trainingConfig, imported));
    } catch (e) {
      set(trainingFailureState(e, "Training checkpoint import failed"));
    }
  },

  verifyTrainingWeightRestoreArtifact: async () => {
    const restorePlan = get().trainingWeightRestorePlan;
    if (restorePlan === null) {
      set(trainingPreconditionErrorState("No training weight restore plan is available."));
      return;
    }
    set(trainingWeightRestoreVerificationStartState());
    try {
      const payload = await fetchStudioJobArtifact(
        restorePlan.source_job_id,
        restorePlan.weights_artifact.relative_path,
      );
      const verification = await verifyTrainingWeightArtifactBlob(restorePlan, payload);
      set(trainingWeightRestoreVerificationLoadedState(verification));
    } catch (error: unknown) {
      set(trainingFailureState(error, "Training weight artifact verification failed"));
    }
  },

  exportTrainingWeightRestoreVerification: () => {
    const { trainingWeightRestorePlan, trainingWeightRestoreVerification } = get();
    const plan = trainingWeightRestoreVerificationExportPlan(
      trainingWeightRestorePlan,
      trainingWeightRestoreVerification,
    );
    if (!plan.available) {
      set(trainingPreconditionErrorState(plan.message));
      return;
    }
    try {
      plan.writeExport();
      set(trainingExportSuccessState());
    } catch (error: unknown) {
      set(trainingFailureState(error, "Training weight restore verification export failed"));
    }
  },

  materializeTrainingWeights: async () => {
    const s = get();
    if (!s.trainingJobId) {
      set(trainingPreconditionErrorState("No completed training job is available."));
      return;
    }
    try {
      const materialization = await apiRestoreTrainingWeights(s.trainingJobId);
      set(trainingWeightMaterializationLoadedState(materialization));
    } catch (error: unknown) {
      set(trainingFailureState(error, "Training weight materialization failed"));
    }
  },

  attachTrainingWeights: async () => {
    const s = get();
    if (!s.trainingJobId) {
      set(trainingPreconditionErrorState("No completed training job is available."));
      return;
    }
    try {
      const attach = await apiAttachTrainingWeights(s.trainingJobId, s.trainingConfig);
      set(trainingWeightAttachLoadedState(attach));
    } catch (error: unknown) {
      set(trainingFailureState(error, "Training weight attach failed"));
    }
  },

  liveAttachTrainingWeights: async () => {
    const s = get();
    const sourceJobId = s.trainingWeightMaterialization?.source_job_id;
    if (!s.trainingJobId || !sourceJobId) {
      set(
        trainingPreconditionErrorState(
          "A running target job and a verified source are required.",
        ),
      );
      return;
    }
    try {
      const attach = await apiAttachTrainingWeightsLive(s.trainingJobId, sourceJobId);
      set(trainingWeightLiveAttachLoadedState(attach));
    } catch (error: unknown) {
      set(trainingFailureState(error, "Training weight live attach failed"));
    }
  },

  setTrainingConfig: (key, value) => {
    set((s) => trainingConfigUpdatedState(s.trainingConfig, key, value));
  },

  resetDefaults: () => {
    const s = get();
    if (s.sourceMode === "model" && s.modelDetail) {
      set({ ...modelDefaultsState(s.modelDetail), ...compilerConfigurationInvalidatedState() });
    }
    void get().runSimulation();
  },

  saveSession: (name) => {
    const s = get();
    const state = studioSavedSessionState(s);
    const nextState = studioSavedSessionUpsertState(s.savedSessions, { name, state });
    set(nextState);
    writeStoredStudioSessions(nextState.savedSessions);
  },

  loadSession: (name) => {
    const session = get().savedSessions.find((ss) => ss.name === name);
    if (!session) return;
    set({
      ...studioSavedSessionRestoreState(session.state),
      ...compilerConfigurationInvalidatedState(),
    });
    void get().runSimulation();
  },

  deleteSession: (name) => {
    const nextState = studioSavedSessionRemovedState(get().savedSessions, name);
    set(nextState);
    writeStoredStudioSessions(nextState.savedSessions);
  },

  applyShareLink: async () => {
    // The Share button has always produced a link. Nothing ever read one back:
    // `readStudioStartupHashState` had no caller outside its own test, so a
    // colleague opening the link got the default Studio and no indication that
    // the link had carried anything. A link handed out that does nothing is
    // worse than no link, because the button promises otherwise.
    //
    // The judgement lives in `studioShareLinkDecision`; this only carries it out.
    const decision = studioShareLinkDecision(
      readStudioStartupHashState(),
      get().models.map((model) => model.name),
    );
    if (decision.kind === "none") return;
    if (decision.kind === "unknown-model") {
      set({ error: decision.message });
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
