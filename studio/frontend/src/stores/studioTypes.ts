// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio Zustand store
// Studio store state shape and view-mode types.

import type {
  CharacterizeResponse, CompileTraceability, FICurveResponse, BifurcationResponse,
  SensitivityResponse, PrecisionResponse, HeatmapResponse, CompareResponse,
  NullclineResponse, FreqResponse, ImportedTrace, NetworkResult, NeuronTemplate,
  ModelSummary, ModelDetail, PresetSummary, SimulateResponse, SynthResult,
  SynthEstimate, MultiTargetResult, SynthToolInfo, SurrogateInfo, TrainingEpochMetrics,
  TrainingWeightRestorePlan, TrainingWeightRestoreResult, TrainingWeightAttachResult,
  TrainingWeightLiveAttachResult, PopulationNode, ProjectionEdge, GraphSimResult,
  NIRFormat, ProjectSaveResponse, ProjectSummary, PipelineResult, StudioAuditExport,
  StudioAuditStatus, StudioCapability, StudioAuditQuarantineArchivePurgeResult,
  StudioAuditQuarantineArchiveResult, StudioAuditQuarantineArchiveRetentionPlan,
  StudioAuditQuarantineArchiveRestoreResult, StudioAuditQuarantineArchiveValidation,
  StudioAuthSession, StudioEvidenceBundleRequest, StudioEvidenceBundleResponse,
  StudioIdentityBrowserUser, StudioIdentityBrowserUserCreate, StudioIdentityServiceAccount,
  StudioJobRecord, StudioJobStatus, StudioOperatorStatus,
} from "../api/client";
import type { StudioSavedSession } from "../studioSavedSessions";
import type { StudioProjectTrainingConfig } from "../studioProjectState";
import type { StudioNetworkParams } from "../studioInputState";
import type { EvidenceBundleSurface } from "../evidenceBundles";
import type { TrainingWeightRestoreVerification } from "../trainingRestore";

export type SourceMode = "model" | "ode";
export type ViewTab = "trace" | "phase" | "isi" | "fi-curve" | "bifurcation" |
  "sensitivity" | "precision" | "heatmap" | "verilog" | "code" |
  "compare" | "freq" | "sta" | "characterize" | "multi" | "network" | "ir" | "synth" | "train" | "canvas" | "delays" | "admin";
export type { EvidenceBundleSurface };

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
  result: SimulateResponse | null;
  fiResult: FICurveResponse | null;
  bifResult: BifurcationResponse | null;
  sensResult: SensitivityResponse | null;
  precResult: PrecisionResponse | null;
  heatmapResult: HeatmapResponse | null;
  compareResult: CompareResponse | null;
  nullclineResult: NullclineResponse | null;
  freqResult: FreqResponse | null;
  staResult: { time_ms: number[]; average: number[]; n_spikes: number } | null;
  charResult: CharacterizeResponse | null;
  multiResults: SimulateResponse[] | null;
  importedTrace: ImportedTrace | null;
  networkResult: NetworkResult | null;
  networkParams: StudioNetworkParams;
  verilogSrc: string;
  irText: string;
  svSource: string;
  irErrors: string[];
  compileTraceability: CompileTraceability | null;
  synthTarget: string;
  synthResult: SynthResult | null;
  synthEstimate: SynthEstimate | null;
  multiTargetResult: MultiTargetResult | null;
  latestSynthesisJobId: string | null;
  latestMultiTargetSynthesisJobId: string | null;
  toolsAvailable: Record<string, SynthToolInfo> | null;
  graphPopulations: PopulationNode[];
  graphProjections: ProjectionEdge[];
  graphModels: string[];
  graphSimResult: GraphSimResult | null;
  progressPct: number;
  progressMsg: string;
  graphErrors: string[];
  projectSaveResult: ProjectSaveResponse | null;
  serverProjects: ProjectSummary[];
  pipelineResult: PipelineResult | null;
  trainingJobId: string | null;
  trainingStatus: string;
  trainingEpochs: TrainingEpochMetrics[];
  trainingWeightRestorePlan: TrainingWeightRestorePlan | null;
  trainingWeightRestoreVerification: TrainingWeightRestoreVerification | null;
  trainingWeightMaterialization: TrainingWeightRestoreResult | null;
  trainingWeightAttach: TrainingWeightAttachResult | null;
  trainingWeightLiveAttach: TrainingWeightLiveAttachResult | null;
  trainingSurrogates: SurrogateInfo[];
  trainingConfig: StudioProjectTrainingConfig;
  codeScript: string;
  codeOneliner: string;
  savedSessions: StudioSavedSession[];
  error: string | null;
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
  setActiveTab: (tab: ViewTab) => void;
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
  runCompile: () => Promise<void>;
  runCharacterize: () => Promise<void>;
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
  runMultiTargetSynthesis: () => Promise<void>;
  runSynthEstimate: () => Promise<void>;
  checkSynthTools: () => Promise<void>;
  saveProjectToServer: (name: string) => Promise<void>;
  loadProjectFromServer: (name: string) => Promise<void>;
  listServerProjects: () => Promise<void>;
  deleteServerProject: (name: string) => Promise<void>;
  runPipelineAction: () => Promise<void>;
  loadGraphModels: () => Promise<void>;
  addPopulation: (neuronType: "excitatory" | "inhibitory") => Promise<void>;
  removePopulation: (id: string) => void;
  updatePopulation: (id: string, updates: Partial<PopulationNode>) => void;
  addProjection: (sourceId: string, targetId: string) => Promise<void>;
  removeProjection: (id: string) => void;
  updateProjection: (id: string, updates: Partial<ProjectionEdge>) => void;
  simulateGraphAction: () => Promise<void>;
  exportGraphNIR: () => Promise<void>;
  importGraphNIR: (nir: NIRFormat) => Promise<void>;
  loadSurrogates: () => Promise<void>;
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
