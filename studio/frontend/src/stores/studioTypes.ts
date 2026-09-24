// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio Zustand store
// Studio store state shape and view-mode types.

import type { StudioGraphHistory } from "../studioGraphHistory";
import type { StudioGraphIssueLocation } from "../studioGraphValidation";
import type {
  StudioTrialMode,
  CharacterizeResponse, CompileTraceability, ModelCosimReport, FICurveResponse, BifurcationResponse,
  SensitivityResponse, PrecisionResponse, HeatmapResponse, CompareResponse,
  NullclineResponse, FreqResponse, ImportedTrace, NetworkResult, NeuronTemplate,
  ModelSummary, ModelDetail, PresetSummary, SimulateResponse, SynthResult,
  SynthEstimate, MultiTargetResult, SynthToolInfo, SurrogateInfo, TrainingEpochMetrics,
  TrainingWeightRestorePlan, TrainingWeightRestoreResult, TrainingWeightAttachResult,
  TrainingWeightLiveAttachResult, PopulationNode, PopulationModelContract, ProjectionEdge, GraphSimResult,
  DeletedProjectSummary, ProjectSaveResponse, ProjectSummary, PipelineResult,
  StudioAuditExport,
  StudioAuditStatus, StudioCapability, StudioAuditQuarantineArchivePurgeResult,
  StudioAuditQuarantineArchiveResult, StudioAuditQuarantineArchiveRetentionPlan,
  StudioAuditQuarantineArchiveRestoreResult, StudioAuditQuarantineArchiveValidation,
  StudioAuthSession, StudioEvidenceBundleRequest, StudioEvidenceBundleResponse,
  StudioIdentityBrowserUser, StudioIdentityBrowserUserCreate, StudioIdentityServiceAccount,
  StudioJobRecord, StudioJobStatus, StudioOperatorStatus, TrainingJobSummary,
} from "../api/client";
import type { StudioSavedSession } from "../studioSavedSessions";
import type {
  StudioCandidateDraft,
  StudioProjectRevisionPointer,
  StudioProjectTrainingConfig,
} from "../studioProjectState";
import type { StudioNetworkParams } from "../studioInputState";
import type { EvidenceBundleSurface } from "../evidenceBundles";
import type { StudioBundleContext } from "../studioBundleContext";
import type { TrainingWeightRestoreVerification } from "../trainingRestore";
import type { GuidedFlowStepKey } from "../guidedFlowState";
import type { StudioStageFailure } from "./studioStageFailure";

/** Whether a run is driven by a catalogue model or by an ODE. */
export type SourceMode = "model" | "ode";
/** Every panel the Studio can show. */
export type ViewTab = "trace" | "phase" | "isi" | "fi-curve" | "bifurcation" |
  "sensitivity" | "precision" | "heatmap" | "verilog" | "code" |
  "compare" | "freq" | "sta" | "characterize" | "multi" | "network" | "ir" | "synth" | "train" | "canvas" | "delays" | "admin" | "candidate";
export type { EvidenceBundleSurface };

/**
 * The whole Studio: every field it holds, and every action that changes one.
 *
 * Data and actions are declared together because that is what a component
 * selects from, and split apart by `StudioStateData` and `StudioStoreActions`
 * below for the two places that need one half without the other.
 */
export interface StudioState {
  sourceMode: SourceMode;
  equations: string[];
  threshold: string;
  reset: string;
  odeParams: Record<string, number>;
  odeInit: Record<string, number>;
  models: ModelSummary[];
  selectedModelName: string;
  modelDetail: ModelDetail | null;
  modelParams: Record<string, number>;
  modelIntegrator: string;
  modelQFormat: string;
  capabilities: StudioCapability[];
  capabilitiesLoading: boolean;
  capabilitiesError: string | null;
  authSession: StudioAuthSession | null;
  authLoading: boolean;
  authError: string | null;
  auditStatus: StudioAuditStatus | null;
  auditExport: StudioAuditExport | null;
  auditArchive: StudioAuditQuarantineArchiveResult | null;
  auditArchiveRetention: StudioAuditQuarantineArchiveRetentionPlan | null;
  auditArchivePurge: StudioAuditQuarantineArchivePurgeResult | null;
  auditArchiveRestore: StudioAuditQuarantineArchiveRestoreResult | null;
  auditArchiveValidation: StudioAuditQuarantineArchiveValidation | null;
  evidenceBundle: StudioEvidenceBundleResponse | null;
  bundleContexts: Partial<Record<EvidenceBundleSurface, StudioBundleContext>>;
  evidenceBundleError: string | null;
  evidenceBundleLoading: boolean;
  projectEvidenceBundle: StudioEvidenceBundleResponse | null;
  projectEvidenceBundleError: string | null;
  projectEvidenceBundleLoading: boolean;
  compileEvidenceBundle: StudioEvidenceBundleResponse | null;
  compileEvidenceBundleError: string | null;
  compileEvidenceBundleLoading: boolean;
  synthesisEvidenceBundle: StudioEvidenceBundleResponse | null;
  synthesisEvidenceBundleError: string | null;
  synthesisEvidenceBundleLoading: boolean;
  jobStatus: StudioJobStatus | null;
  jobRecords: StudioJobRecord[];
  identityBrowserUsers: StudioIdentityBrowserUser[];
  identityServiceAccounts: StudioIdentityServiceAccount[];
  operatorStatus: StudioOperatorStatus | null;
  auditLoading: boolean;
  auditError: string | null;
  templates: NeuronTemplate[];
  presets: PresetSummary[];
  dt: number;
  duration: number;
  current: number;
  protocol: string;
  frequencyHz: number;
  seed: number | null;
  trial: StudioTrialMode;
  result: SimulateResponse | null;
  fiResult: FICurveResponse | null;
  bifResult: BifurcationResponse | null;
  sensResult: SensitivityResponse | null;
  precResult: PrecisionResponse | null;
  heatmapResult: HeatmapResponse | null;
  heatmapExperimentKey: string | null;
  compareResult: CompareResponse | null;
  nullclineResult: NullclineResponse | null;
  freqResult: FreqResponse | null;
  staResult: { time_ms: number[]; average: number[]; n_spikes: number } | null;
  charResult: CharacterizeResponse | null;
  /**
   * The experiment `result` was produced under, or `null` when none has been.
   *
   * A run is evidence only of the configuration that produced it. The guided
   * workflow compares this against the configuration in force, so a trace left
   * over from a model the reader has since replaced stops counting as a
   * completed simulation instead of quietly satisfying the step.
   */
  resultExperimentKey: string | null;
  /**
   * The experiment the most recent analysis was produced under.
   *
   * One key for all nine analyses: the workflow's analyse step asks whether
   * some analysis of the current run exists, not which one.
   */
  analysisExperimentKey: string | null;
  multiResults: SimulateResponse[] | null;
  importedTrace: ImportedTrace | null;
  networkResult: NetworkResult | null;
  networkParams: StudioNetworkParams;
  verilogSrc: string;
  irText: string;
  svSource: string;
  irErrors: string[];
  compileTraceability: CompileTraceability | null;
  cosimResult: ModelCosimReport | null;
  synthTarget: string;
  synthResult: SynthResult | null;
  synthEstimate: SynthEstimate | null;
  multiTargetResult: MultiTargetResult | null;
  latestSynthesisJobId: string | null;
  latestMultiTargetSynthesisJobId: string | null;
  toolsAvailable: Record<string, SynthToolInfo> | null;
  /** Candidate model drafts the workspace holds, each exactly as typed. */
  candidates: StudioCandidateDraft[];
  graphPopulations: PopulationNode[];
  /** Graph edits that can be stepped back and forward; layout moves are not edits. */
  graphHistory: StudioGraphHistory;
  graphProjections: ProjectionEdge[];
  graphModels: string[];
  graphSimResult: GraphSimResult | null;
  progressPct: number;
  progressMsg: string;
  graphErrors: string[];
  /**
   * The same failures, each resolved to the population or projection it is
   * about. Kept beside `graphErrors` rather than replacing it, because a
   * caller that only wants the sentences should not have to locate them.
   */
  graphIssues: StudioGraphIssueLocation[];
  /** Which projection the property editor is editing, if any. */
  selectedProjectionId: string | null;
  /** Which population the property editor is editing, if any. */
  selectedPopulationId: string | null;
  /**
   * Every population the canvas currently has selected.
   *
   * Separate from the editor's single selection: a duplicate operates on a
   * whole group, and the editor on exactly one.
   */
  selectedPopulationIds: string[];
  /**
   * What the last graph operation did, in words, or `null`.
   *
   * A duplicate leaves projections at the boundary of its selection, and that
   * is not visible in the diagram; it has to be said.
   */
  graphNotice: string | null;
  /**
   * The contract of the selected population's model, once it has arrived.
   *
   * Which constructor fields a population may override is the server's answer,
   * so the editor waits for it rather than offering a guess.
   */
  populationModelContract: PopulationModelContract | null;
  projectSaveResult: ProjectSaveResponse | null;
  /** The workspace revision the editor loaded or last wrote. */
  projectRevision: StudioProjectRevisionPointer | null;
  /** Workspaces waiting in the recoverable trash, newest first. */
  deletedProjects: DeletedProjectSummary[];
  serverProjects: ProjectSummary[];
  pipelineResult: PipelineResult | null;
  trainingJobId: string | null;
  trainingStatus: string;
  trainingEpochs: TrainingEpochMetrics[];
  trainingJobs: TrainingJobSummary[];
  trainingJobsLoading: boolean;
  trainingJobsError: string | null;
  /** The retained run's configuration, separate from editable project settings. */
  trainingObservedConfig: StudioProjectTrainingConfig | null;
  /**
   * The experiment the current training run was started under.
   *
   * Training is the one step whose artefact outlives its configuration on
   * purpose -- weights are restored into later runs -- so completion is bound
   * to the configuration that trained them rather than inferred from their
   * presence.
   */
  trainingExperimentKey: string | null;
  trainingWeightRestorePlan: TrainingWeightRestorePlan | null;
  trainingWeightRestoreVerification: TrainingWeightRestoreVerification | null;
  trainingWeightMaterialization: TrainingWeightRestoreResult | null;
  trainingWeightAttach: TrainingWeightAttachResult | null;
  trainingWeightLiveAttach: TrainingWeightLiveAttachResult | null;
  trainingSurrogates: SurrogateInfo[];
  trainingConfig: StudioProjectTrainingConfig;
  codeScript: string;
  codeOneliner: string;
  /** Script that replays a saved pack and compares the run in full. */
  codeReplayScript: string;
  /** Digest of the exported effective experiment. */
  codeExperimentSha256: string;
  savedSessions: StudioSavedSession[];
  error: string | null;
  /** The latest failed attempt at a workflow stage, with the experiment it failed under. */
  stageFailure: StudioStageFailure | null;
  /**
   * Where a refused edit diverged from, set only by a save conflict.
   * `null` means there is nothing waiting to be kept.
   */
  refusedEdit: { name: string; baseRevision: number } | null;
  isSimulating: boolean;
  activeTab: ViewTab;
  modelFilter: string;
  sweepParam: string;

  setSourceMode: (m: SourceMode) => void;
  setEquations: (eqs: string[]) => void;
  setThreshold: (t: string) => void;
  setReset: (r: string) => void;
  setOdeParam: (key: string, value: number) => void;
  setOdeInit: (key: string, value: number) => void;
  setModelParam: (key: string, value: number) => void;
  setModelIntegrator: (integrator: string) => void;
  setModelQFormat: (qFormat: string) => void;
  setDt: (dt: number) => void;
  setDuration: (d: number) => void;
  setCurrent: (c: number) => void;
  setProtocol: (p: string) => void;
  setFrequencyHz: (frequencyHz: number) => void;
  setSeed: (seed: number | null) => void;
  setTrial: (trial: StudioTrialMode) => void;
  setActiveTab: (tab: ViewTab) => void;
  /** Replace the workspace's candidate draft; an empty draft removes it. */
  setCandidateDraft: (text: string) => void;
  setModelFilter: (f: string) => void;
  setSweepParam: (p: string) => void;

  loadTemplates: () => Promise<void>;
  loadModels: () => Promise<void>;
  loadPresets: () => Promise<void>;
  loadCapabilities: () => Promise<void>;
  loadAuthSession: () => Promise<void>;
  loginBrowserUser: (username: string, password: string) => Promise<void>;
  logoutBrowserUser: () => Promise<void>;
  loadAuditStatus: () => Promise<void>;
  loadAuditExport: () => Promise<void>;
  createAuditQuarantineArchive: (limit: number) => Promise<void>;
  validateAuditQuarantineArchive: (
    archive: Record<string, unknown>,
    manifest: Record<string, unknown> | null,
  ) => Promise<void>;
  loadAuditQuarantineArchiveRetention: (retainLatest: number) => Promise<void>;
  restoreAuditQuarantineArchive: (
    archive: Record<string, unknown>,
    manifest: Record<string, unknown> | null,
  ) => Promise<void>;
  purgeAuditQuarantineArchiveRetention: (retainLatest: number) => Promise<void>;
  createEvidenceBundle: (request: StudioEvidenceBundleRequest) => Promise<void>;
  createEvidenceBundleForSurface: (
    surface: EvidenceBundleSurface,
    request: StudioEvidenceBundleRequest,
  ) => Promise<void>;
  downloadEvidenceBundleArtifact: (relativePath: string) => Promise<void>;
  downloadEvidenceBundleArtifactForSurface: (
    surface: EvidenceBundleSurface,
    relativePath: string,
  ) => Promise<void>;
  loadJobStatus: () => Promise<void>;
  loadIdentityServiceAccounts: () => Promise<void>;
  createIdentityBrowserUser: (create: StudioIdentityBrowserUserCreate) => Promise<void>;
  updateIdentityServiceAccount: (
    principalId: string,
    update: { active: boolean; expires_at_utc: string | null; roles: string[] },
  ) => Promise<void>;
  updateIdentityBrowserUser: (
    username: string,
    update: { active: boolean; expires_at_utc: string | null; roles: string[] },
  ) => Promise<void>;
  rotateIdentityBrowserUserPassword: (username: string, password: string) => Promise<void>;
  loadOperatorStatus: () => Promise<void>;
  selectTemplate: (name: string) => void;
  selectModel: (name: string) => Promise<void>;
  loadPreset: (id: string) => Promise<void>;
  runSimulation: () => Promise<void>;
  runFICurve: () => Promise<void>;
  runBifurcation: () => Promise<void>;
  runSensitivity: () => Promise<void>;
  runPrecision: () => Promise<void>;
  runHeatmap: () => Promise<void>;
  runCodegen: () => Promise<void>;
  exportReplayPack: () => Promise<void>;
  runCompile: () => Promise<void>;
  runCosim: () => Promise<void>;
  runCharacterize: () => void;
  runMultiSimulate: (modelNames: string[]) => Promise<void>;
  runNetwork: () => Promise<void>;
  setNetworkParam: <K extends keyof StudioNetworkParams>(
    key: K,
    value: StudioNetworkParams[K],
  ) => void;
  importCSV: (csv: string) => Promise<void>;
  runCompare: (configB: Record<string, unknown>) => Promise<void>;
  runNullclines: () => Promise<void>;
  runFreqResponse: () => Promise<void>;
  computeSTA: () => void;
  runBuildIR: () => Promise<void>;
  runEmitSV: () => Promise<void>;
  setSynthTarget: (t: string) => void;
  runSynthesis: () => Promise<void>;
  /** Record how a guided attempt at a stage ended: its failure, or `null` on success. */
  recordStageOutcome: (stage: GuidedFlowStepKey, failure: string | null) => void;
  runMultiTargetSynthesis: () => Promise<void>;
  runSynthEstimate: () => Promise<void>;
  checkSynthTools: () => Promise<void>;
  saveProjectToServer: (name: string) => Promise<void>;
  /**
   * Store the edit a save conflict refused, as its own branch.
   *
   * A no-op unless a conflict has set `refusedEdit`, which is the only
   * thing that can: there is nothing to keep otherwise.
   */
  keepRefusedEdit: () => Promise<void>;
  /**
   * Apply the workspace a share link was opened with.
   *
   * A no-op without a link. When the link names a model this catalogue does
   * not hold — the corpus renames identities — it says so rather than
   * selecting nothing.
   */
  applyShareLink: () => Promise<void>;
  loadProjectFromServer: (name: string, revision?: number | null) => Promise<void>;
  listServerProjects: () => Promise<void>;
  deleteServerProject: (name: string) => Promise<void>;
  listDeletedServerProjects: () => Promise<void>;
  restoreDeletedServerProject: (token: string) => Promise<void>;
  runPipelineAction: () => Promise<void>;
  loadGraphModels: () => Promise<void>;
  addPopulation: (neuronType: "excitatory" | "inhibitory") => Promise<void>;
  removePopulation: (id: string) => void;
  undoGraphEdit: () => void;
  redoGraphEdit: () => void;
  updatePopulation: (id: string, updates: Partial<PopulationNode>) => void;
  addProjection: (sourceId: string, targetId: string) => Promise<void>;
  removeProjection: (id: string) => void;
  updateProjection: (id: string, updates: Partial<ProjectionEdge>) => void;
  /** Select the projection the property editor edits, or clear the selection. */
  selectProjection: (id: string | null) => void;
  /** Select the population the property editor edits, or clear the selection. */
  selectPopulation: (id: string | null) => void;
  /** Record the canvas's current multi-selection of populations. */
  selectPopulations: (ids: string[]) => void;
  /**
   * Duplicate the selected populations and the projections between them.
   *
   * A projection with one end outside the selection is not copied; the notice
   * says how many were left behind.
   */
  duplicateSelection: () => Promise<void>;
  /** Fetch the contract of one model, so its parameters can be edited. */
  loadPopulationModelContract: (model: string) => Promise<void>;
  /**
   * Ask the server whether the graph is admissible as it now stands.
   *
   * Separate from running it: an editor has to be able to check a change
   * without starting a simulation, and a graph the server accepts must clear
   * the previous refusal rather than leave it beside the fixed field.
   */
  validateGraphAction: () => Promise<void>;
  simulateGraphAction: () => Promise<void>;
  /** Write the canvas network as a NIR file and say what the file does not carry. */
  exportGraphNIR: () => Promise<void>;
  /** Replace the canvas with the network in a NIR file or a saved graph envelope. */
  importGraphNIR: (file: Blob & { name: string }) => Promise<void>;
  loadSurrogates: () => Promise<void>;
  loadTrainingJobs: () => Promise<void>;
  selectTrainingJob: (jobId: string) => Promise<void>;
  startTraining: () => Promise<void>;
  stopTraining: () => Promise<void>;
  exportTrainingCheckpoint: () => Promise<void>;
  importTrainingCheckpointText: (checkpointJson: string) => Promise<void>;
  verifyTrainingWeightRestoreArtifact: () => Promise<void>;
  exportTrainingWeightRestoreVerification: () => void;
  materializeTrainingWeights: () => Promise<void>;
  attachTrainingWeights: () => Promise<void>;
  liveAttachTrainingWeights: () => Promise<void>;
  setTrainingConfig: <K extends keyof StudioProjectTrainingConfig>(
    key: K,
    value: StudioProjectTrainingConfig[K],
  ) => void;
  autoSimulate: () => void;
  exportData: () => void;
  exportCSV: () => void;
  exportSVG: () => void;
  resetDefaults: () => void;
  saveSession: (name: string) => void;
  loadSession: (name: string) => void;
  deleteSession: (name: string) => void;
  shareURL: () => void;
  sweepParamY: string;
  setSweepParamY: (p: string) => void;
}

/**
 * The store's action half: every field of the state that is callable.
 *
 * Derived rather than written out, so an action added to `StudioState` is
 * automatically required of the actions factory instead of quietly becoming
 * optional. Before this existed the factory returned `Partial<StudioState>`
 * and the store was assembled through a cast, which meant a forgotten action
 * was a runtime `undefined` rather than a type error.
 */
export type StudioStoreActions = {
  [K in keyof StudioState as StudioState[K] extends (...args: never[]) => unknown
    ? K
    : never]: StudioState[K];
};

/** The store's data half: everything the state holds that is not an action. */
export type StudioStateData = Omit<StudioState, keyof StudioStoreActions>;
