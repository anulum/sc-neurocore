// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio Zustand store
// Studio store action implementations (mutations + async side effects).

import {
  fetchTemplates,
  fetchModels,
  fetchModelDetail,
  fetchPresets,
  fetchPreset,
  fetchStudioAuthSession,
  createStudioAuditQuarantineArchive,
  createStudioEvidenceBundle,
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
  compileVerilog,
  compileModelVerilog,
  fetchPrecision,
  fetchCodegen,
  fetchCompare,
  fetchNullclines,
  fetchFreqResponse,
  fetchCharacterize,
  fetchMultiSimulate,
  importTrace,
  simulateNetwork,
  buildIR,
  emitSV,
  emitSVDirect,
  fetchSynthTools,
  runSynthesis as apiRunSynthesis,
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
  validateGraph as apiValidateGraph,
  exportNIR as apiExportNIR,
  importNIR as apiImportNIR,
  saveProject as apiSaveProject,
  loadProject as apiLoadProject,
  listProjects as apiListProjects,
  deleteProject as apiDeleteProject,
  runPipeline as apiRunPipeline,
  createStudioIdentityBrowserUser,
  setStudioAuthToken,
  updateStudioIdentityBrowserUser,
  updateStudioIdentityServiceAccount,
  validateStudioAuditQuarantineArchive,
  connectProgress,
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
  studioProjectFailureState,
  studioProjectListLoadedState,
  studioProjectRestoreState,
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
  studioGraphValidationFailedState,
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
  copyStudioShareUrlInRuntime,
  scheduleStudioShareStatusClear,
  studioShareStatusClearedState,
  studioShareStatusState,
} from "../studioShareRuntime";
import {
  studioTraceImportRequest,
} from "../studioTraceImport";
import {
  studioCodegenRequest,
  studioFrequencyResponseRequest,
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
  simulationExportPlan,
} from "../simulationExports";
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
  adminEvidenceBundleCreatedState,
  adminEvidenceBundleFailureState,
  adminEvidenceBundleLoadingState,
  evidenceBundleArtifactDownloadPlan,
  scopedEvidenceBundleCreatedState,
  scopedEvidenceBundleFailureState,
  scopedEvidenceBundleLoadingState,
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
  multiTargetSynthesisRunStartState,
  synthesisErrorMessageState,
  synthesisErrorState,
  synthesisEstimateLoadedState,
  synthesisFailureState,
  synthesisRunCompletedState,
  synthesisRunStartState,
  synthesisTargetState,
  synthesisToolStatusLoadedState,
} from "../synthesisStoreState";
import {
  compilerErrorState,
  compilerFailureState,
  compilerIRLoadedState,
  compilerRunStartState,
  compilerSVDirectLoadedState,
  compilerSVLoadedState,
  compilerVerilogLoadedState,
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
import {
  characterizeCompleteState,
  characterizeFailureState,
  characterizeProgressMessageState,
  characterizeRequestConfig,
  characterizeRunStartState,
} from "../characterizeStoreState";
import {
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
import type { StudioState } from "./studioTypes";
import {
  runStoreHeavyAnalysis,
  simulationConfigInput,
} from "./studioHeavyAnalysis";
import {
  scheduleStudioAutoSimulation,
  type StudioAutoSimulationTimer,
} from "../studioAutoSimulation";
import { modelCompileRequest } from "../modelCompileConfig";

let debounceTimer: StudioAutoSimulationTimer | null = null;

export function createStudioStoreActions(
  set: (partial: Partial<StudioState> | ((state: StudioState) => Partial<StudioState>)) => void,
  get: () => StudioState,
): Partial<StudioState> {
  return {
  setSourceMode: (m) => set(sourceModeState(m)),
  setEquations: (eqs) => { set(equationsState(eqs)); get().autoSimulate(); },
  setThreshold: (t) => { set(thresholdState(t)); get().autoSimulate(); },
  setReset: (r) => { set(resetState(r)); get().autoSimulate(); },
  setOdeParam: (key, value) => {
    set((s) => numberRecordEntryState("odeParams", s.odeParams, key, value));
    get().autoSimulate();
  },
  setOdeInit: (key, value) => {
    set((s) => numberRecordEntryState("odeInit", s.odeInit, key, value));
    get().autoSimulate();
  },
  setModelParam: (key, value) => {
    set((s) => numberRecordEntryState("modelParams", s.modelParams, key, value));
    get().autoSimulate();
  },
  setModelIntegrator: (modelIntegrator) => set({ modelIntegrator }),
  setModelQFormat: (modelQFormat) => set({ modelQFormat }),
  setDt: (dt) => { set(dtState(dt)); get().autoSimulate(); },
  setDuration: (d) => { set(durationState(d)); get().autoSimulate(); },
  setCurrent: (c) => { set(currentState(c)); get().autoSimulate(); },
  setProtocol: (p) => { set(protocolState(p)); get().autoSimulate(); },
  setActiveTab: (tab) => set(activeTabState(tab)),
  setModelFilter: (f) => set(modelFilterState(f)),
  setSweepParam: (p) => set(sweepParamState(p)),
  setSweepParamY: (p) => set(sweepParamYState(p)),

  loadTemplates: async () => set(templatesLoadedState(await fetchTemplates())),
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
  createEvidenceBundle: async (request) => {
    set(adminEvidenceBundleLoadingState());
    try {
      const evidenceBundle = await createStudioEvidenceBundle(request);
      const [operatorStatus, jobList] = await Promise.all([
        fetchStudioOperatorStatus(),
        fetchStudioJobs(),
      ]);
      set(adminEvidenceBundleCreatedState(evidenceBundle, operatorStatus, jobList));
    } catch (error: unknown) {
      set(adminEvidenceBundleFailureState(error));
    }
  },
  createEvidenceBundleForSurface: async (surface, request) => {
    if (surface === "admin") {
      await get().createEvidenceBundle(request);
      return;
    }
    set(scopedEvidenceBundleLoadingState(surface));
    try {
      const evidenceBundle = await createStudioEvidenceBundle(request);
      const [operatorStatus, jobList] = await Promise.all([
        fetchStudioOperatorStatus(),
        fetchStudioJobs(),
      ]);
      set(scopedEvidenceBundleCreatedState(surface, evidenceBundle, operatorStatus, jobList));
    } catch (error: unknown) {
      set(scopedEvidenceBundleFailureState(surface, error));
    }
  },
  downloadEvidenceBundleArtifact: async (relativePath) => {
    await get().downloadEvidenceBundleArtifactForSurface("admin", relativePath);
  },
  downloadEvidenceBundleArtifactForSurface: async (surface, relativePath) => {
    const downloadPlan = evidenceBundleArtifactDownloadPlan(surface, relativePath, get());
    if (!downloadPlan.available) {
      set(downloadPlan.statePatch);
      return;
    }
    set(downloadPlan.startState);
    try {
      const payload = await fetchStudioJobArtifact(downloadPlan.jobId, downloadPlan.relativePath);
      downloadPlan.writePayload(payload);
    } catch (error: unknown) {
      set(downloadPlan.failureState(error));
    }
  },
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
    if (models.length > 0 && !get().selectedModelName) await get().selectModel(models[0].name);
  },
  loadPresets: async () => set(presetsLoadedState(await fetchPresets())),

  selectTemplate: (name) => {
    const template = get().templates.find((candidate) => candidate.name === name);
    if (template === undefined) return;
    set(templateSelectedState(template));
    get().runSimulation();
  },

  selectModel: async (name) => {
    set(modelSelectionStartedState(name));
    const detail = await fetchModelDetail(name);
    if (detail === null) return;
    set(modelDetailLoadedState(detail));
    get().runSimulation();
  },

  loadPreset: async (id) => {
    const preset = await fetchPreset(id);
    const selection = presetSelection(preset);
    if (selection.modelName !== null) {
      await get().selectModel(selection.modelName);
      if (selection.modelRuntimeState !== null) set(selection.modelRuntimeState);
    } else if (selection.odeState !== null) {
      set(selection.odeState);
    }
    if (selection.action.kind === "fi-curve") get().runFICurve();
    else if (selection.action.kind === "precision") get().runPrecision();
    else {
      set(activeTabState(selection.action.activeTab));
      get().runSimulation();
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
    set(studioAnalysisStartState());
    try {
      const cfg = studioSimulationConfig(simulationConfigInput(s));
      const result = s.sourceMode === "model" && s.selectedModelName
        ? await simulateModel(cfg) : await simulateODE(cfg);
      set(studioSimulationResultState(result));
    } catch (e) {
      set(studioAnalysisFailureState(e));
    }
  },

  runFICurve: () => runStoreHeavyAnalysis("fi_curve", get, set),
  runBifurcation: () => runStoreHeavyAnalysis("bifurcation", get, set),
  runSensitivity: () => runStoreHeavyAnalysis("sensitivity", get, set),

  runPrecision: async () => {
    const s = get();
    if (s.sourceMode !== "ode") {
      set(studioAnalysisErrorState("Precision compare only for custom ODE mode"));
      return;
    }
    set(studioAnalysisStartState("precision"));
    try {
      const precResult = await fetchPrecision(studioPrecisionRequest(simulationConfigInput(s)));
      set(studioPrecisionResultState(precResult));
    } catch (e) { set(studioAnalysisFailureState(e)); }
  },

  runCompile: async () => {
    const s = get();
    set(compilerRunStartState("verilog"));
    try {
      const res = s.sourceMode === "model"
        ? await compileModelVerilog(modelCompileRequest({
          dt: s.dt,
          integrator: s.modelIntegrator,
          modelDetail: s.modelDetail,
          modelParams: s.modelParams,
          qFormat: s.modelQFormat,
          selectedModelName: s.selectedModelName,
        }))
        : await compileVerilog({
          equations: s.equations, threshold: s.threshold, reset: s.reset, params: s.odeParams,
        });
      set(compilerVerilogLoadedState(res));
    } catch (e) { set(compilerFailureState(e)); }
  },

  runHeatmap: () => runStoreHeavyAnalysis("heatmap", get, set),

  runCodegen: async () => {
    const s = get();
    set(studioCodegenStartState());
    try {
      const res = await fetchCodegen(studioCodegenRequest(simulationConfigInput(s)));
      set(studioCodegenResultState(res.script, res.oneliner));
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

  runCharacterize: async () => {
    const s = get();
    if (s.isSimulating || !s.selectedModelName) return;
    set(characterizeRunStartState());
    const config = characterizeRequestConfig(s);
    const ws = connectProgress("characterize", config, (msg) => {
      const state = characterizeProgressMessageState(msg);
      if (state !== null) set(state);
    });
    ws.onerror = () => {
      fetchCharacterize(config).then(
        (charResult) => set(characterizeCompleteState(charResult)),
        (e) => set(characterizeFailureState(e)),
      );
    };
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

  runCompare: async (configB) => {
    const s = get();
    if (s.isSimulating) return;
    set(studioAnalysisStartState("compare"));
    try {
      const configA = studioSimulationConfig(simulationConfigInput(s));
      const compareResult = await fetchCompare(configA, configB);
      set(studioCompareResultState(compareResult));
    } catch (e) { set(studioAnalysisFailureState(e)); }
  },

  runNullclines: async () => {
    const s = get();
    if (s.sourceMode !== "ode" || s.equations.length < 2) {
      set(studioAnalysisErrorState("Nullclines need 2+ variable ODE in custom mode"));
      return;
    }
    set(studioAnalysisStartState());
    try {
      const vars = Object.keys(s.odeInit);
      if (vars.length < 2) { set(studioAnalysisIdleState()); return; }
      const v0vals = s.result?.states[vars[0]];
      const v1vals = s.result?.states[vars[1]];
      const r0: [number, number] = v0vals
        ? [Math.min(...v0vals) - 10, Math.max(...v0vals) + 10]
        : [-80, 40];
      const r1: [number, number] = v1vals
        ? [Math.min(...v1vals) - 0.5, Math.max(...v1vals) + 0.5]
        : [-2, 2];
      const nullclineResult = await fetchNullclines({
        equations: s.equations, params: s.odeParams,
        var_names: vars, ranges: { [vars[0]]: r0, [vars[1]]: r1 }, grid_size: 60,
      });
      set(studioNullclineResultState(nullclineResult));
    } catch (e) { set(studioAnalysisFailureState(e)); }
  },

  runFreqResponse: async () => {
    const s = get();
    if (s.isSimulating) return;
    set(studioAnalysisStartState("freq"));
    try {
      const cfg = studioSimulationConfig(simulationConfigInput(s));
      const freqResult = await fetchFreqResponse(studioFrequencyResponseRequest(cfg, s.current));
      set(studioFrequencyResultState(freqResult));
    } catch (e) { set(studioAnalysisFailureState(e)); }
  },

  computeSTA: () => {
    const { result } = get();
    if (!result) return;
    const state = studioSTAResultState(result);
    if (state !== null) set(state);
  },

  runBuildIR: async () => {
    const s = get();
    if (s.sourceMode !== "ode") { set(compilerErrorState("IR build requires ODE mode")); return; }
    set(compilerRunStartState("ir"));
    try {
      const cfg = {
        equations: s.equations, threshold: s.threshold || null, reset: s.reset || null,
        params: s.odeParams, dt: s.dt,
      };
      const result = await buildIR(cfg);
      set(compilerIRLoadedState(result));
      if (result.errors.length === 0) {
        const sv = await emitSV(result.ir_text);
        set(compilerSVLoadedState(sv));
      }
    } catch (e) { set(compilerFailureState(e)); }
  },

  runEmitSV: async () => {
    const s = get();
    if (s.sourceMode !== "ode") { set(compilerErrorState("SV emit requires ODE mode")); return; }
    set(compilerRunStartState("ir"));
    try {
      const result = await emitSVDirect({
        equations: s.equations, threshold: s.threshold || null, reset: s.reset || null,
        params: s.odeParams,
      });
      set(compilerSVDirectLoadedState(result));
    } catch (e) { set(compilerFailureState(e)); }
  },

  setSynthTarget: (t) => set(synthesisTargetState(t)),

  runSynthesis: async () => {
    const s = get();
    if (!s.svSource && !s.verilogSrc) {
      set(synthesisErrorMessageState("Generate Verilog first"));
      return;
    }
    set(synthesisRunStartState());
    try {
      const verilog = s.svSource || s.verilogSrc;
      const synthResult = await apiRunSynthesis(verilog, s.synthTarget);
      const [operatorStatus, jobList] = await Promise.all([
        fetchStudioOperatorStatus(),
        fetchStudioJobs(),
      ]);
      set(synthesisRunCompletedState(synthResult, operatorStatus, jobList));
    } catch (e) { set(synthesisFailureState(e)); }
  },

  runMultiTargetSynthesis: async () => {
    const s = get();
    if (!s.svSource && !s.verilogSrc) {
      set(synthesisErrorMessageState("Generate Verilog first"));
      return;
    }
    set(multiTargetSynthesisRunStartState());
    try {
      const verilog = s.svSource || s.verilogSrc;
      const multiTargetResult = await runMultiTargetSynthesis(verilog);
      const [operatorStatus, jobList] = await Promise.all([
        fetchStudioOperatorStatus(),
        fetchStudioJobs(),
      ]);
      set(multiTargetSynthesisRunCompletedState(multiTargetResult, operatorStatus, jobList));
    } catch (e) { set(synthesisFailureState(e)); }
  },

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
    try {
      const projectSaveResult = await apiSaveProject(name, state);
      set(studioProjectSavedState(projectSaveResult));
      await get().listServerProjects();
    } catch (e) { set(studioProjectFailureState(e, "Project save failed")); }
  },

  loadProjectFromServer: async (name) => {
    try {
      const data = await apiLoadProject(name);
      const projectState = studioProjectStateFromLoadResponse(data, get().trainingConfig);
      set(studioProjectRestoreState(projectState));
      get().runSimulation();
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
    } catch (e) { set(studioProjectFailureState(e, "Project delete failed")); }
  },

  runPipelineAction: async () => {
    const s = get();
    if (s.isSimulating || s.graphPopulations.length === 0) return;
    set(studioPipelineStartState());
    try {
      const graph = studioGraphRequest(s.graphPopulations, s.graphProjections, s.duration, s.dt);
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
      set((prev) => studioPopulationAddedState(prev.graphPopulations, pop));
    } catch (e) { set(studioGraphFailureState(e, "Population creation failed")); }
  },

  removePopulation: (id) => {
    set((s) => {
      const graph = studioGraphWithoutPopulation({
        populations: s.graphPopulations,
        projections: s.graphProjections,
      }, id);
      return {
        graphPopulations: graph.populations,
        graphProjections: graph.projections,
      };
    });
  },

  updatePopulation: (id, updates) => {
    set((s) => studioPopulationUpdatedState(s.graphPopulations, id, updates));
  },

  addProjection: async (sourceId, targetId) => {
    try {
      const proj = await apiCreateProj(studioDefaultProjectionRequest(sourceId, targetId));
      set((prev) => studioProjectionAddedState(prev.graphProjections, proj));
    } catch (e) { set(studioGraphFailureState(e, "Projection creation failed")); }
  },

  removeProjection: (id) => {
    set((s) => studioProjectionRemovedState(s.graphProjections, id));
  },

  updateProjection: (id, updates) => {
    set((s) => studioProjectionUpdatedState(s.graphProjections, id, updates));
  },

  simulateGraphAction: async () => {
    const s = get();
    if (s.isSimulating) return;
    set(studioGraphSimulationStartState());
    try {
      const graph = studioGraphRequest(s.graphPopulations, s.graphProjections, s.duration, s.dt);
      const validation = await apiValidateGraph(graph);
      if (!validation.valid) {
        set(studioGraphValidationFailedState(validation.errors));
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
    set(trainingStartState());
    try {
      const result = await apiStartTraining(s.trainingConfig);
      set(trainingStartedState(result.job_id));
      connectStudioTrainingEventSource(result.job_id, {
        onDisconnected: () => set(trainingStreamDisconnectedState()),
        onEpoch: (metrics) =>
          set((prev) => trainingEpochAppendedState(prev.trainingEpochs, metrics)),
        onError: (message) => set(trainingStreamErrorState(message)),
        onTerminal: (status) => set(trainingTerminalState(status)),
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
      set(modelDefaultsState(s.modelDetail));
    }
    get().runSimulation();
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
    set(studioSavedSessionRestoreState(session.state));
    get().runSimulation();
  },

  deleteSession: (name) => {
    const nextState = studioSavedSessionRemovedState(get().savedSessions, name);
    set(nextState);
    writeStoredStudioSessions(nextState.savedSessions);
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
      scheduleStudioShareStatusClear(() => set(studioShareStatusClearedState()));
    });
  },
  };
}
