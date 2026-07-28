// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio frontend API
// Public Studio API facade — re-exports transport, DTOs, and domain endpoints.

export {
  setStudioAuthToken,
  progressWebSocketProtocols,
} from "./http";

export type * from "./types";

export {
  fetchTemplates,
  fetchModels,
  fetchModelDetail,
  fetchModelFacets,
  fetchModelDoc,
  fetchPresets,
  fetchPreset,
  fetchModelScan,
  submitModelScanJob,
} from "./modelsApi";

export {
  fetchDclsInfo,
  fetchDclsBenchmark,
  evaluateDcls,
} from "./dclsApi";

export {
  runBenchmark,
  contributeBenchmark,
  fetchDatabank,
} from "./benchmarksApi";

export {
  fetchStudioCapabilities,
  fetchStudioAuditStatus,
  fetchStudioAuditExport,
  createStudioAuditQuarantineArchive,
  validateStudioAuditQuarantineArchive,
  fetchStudioAuditQuarantineArchiveRetention,
  restoreStudioAuditQuarantineArchive,
  purgeStudioAuditQuarantineArchiveRetention,
  fetchStudioJobStatus,
  fetchStudioJobs,
  fetchStudioJobRecord,
  fetchStudioJobAtStatusRoute,
  fetchStudioJobArtifact,
  createStudioEvidenceBundle,
  fetchStudioOperatorStatus,
  fetchStudioIdentityServiceAccounts,
  fetchStudioIdentityBrowserUsers,
  createStudioIdentityBrowserUser,
  updateStudioIdentityServiceAccount,
  updateStudioIdentityBrowserUser,
  rotateStudioIdentityBrowserUserPassword,
  loginStudioBrowserUser,
  fetchStudioAuthSession,
  logoutStudioBrowserUser,
} from "./adminApi";

export {
  simulateODE,
  simulateModel,
  simulateNetwork,
  fetchCharacterize,
  fetchMultiSimulate,
  importTrace,
} from "./simulationApi";

export {
  submitAnalysisJob,
  fetchNullclines,
  fetchPrecision,
  fetchCompare,
  fetchFreqResponse,
  fetchCodegen,
} from "./analysisApi";

export {
  compileVerilog,
  compileModelVerilog,
  cosimModelVerilog,
  buildIR,
  verifyIR,
  emitSV,
  emitSVDirect,
  fetchCosimDetail,
} from "./compilerApi";

export {
  fetchSynthTools,
  runSynthesis,
  runMultiTargetSynthesis,
  fetchSynthEstimate,
  runPnR,
} from "./synthApi";

export {
  fetchSurrogates,
  fetchCellTypes,
  startTraining,
  stopTraining,
  fetchTrainingStatus,
  fetchTrainingJobs,
  exportTrainingCheckpoint,
  importTrainingCheckpoint,
  restoreTrainingWeights,
  attachTrainingWeights,
  attachTrainingWeightsLive,
} from "./trainingApi";

export {
  fetchGraphModels,
  createPopulation,
  createProjection,
  validateGraph,
  simulateGraph,
  exportNIR,
  importNIR,
} from "./graphApi";

export {
  saveProject,
  loadProject,
  listProjects,
  deleteProject,
  runPipeline,
} from "./projectApi";

export {
  connectProgress,
} from "./progressApi";

// Legacy sync analysis helpers fetchFICurve/fetchBifurcation/fetchSensitivity/fetchHeatmap
// removed: zero callers; async path is submitAnalysisJob + job polling (W12-D/G).
