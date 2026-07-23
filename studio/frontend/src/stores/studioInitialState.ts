// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio Zustand store
// Default initial field values for the Studio Zustand store.

import { readStoredStudioSessions } from "../studioSavedSessions";

/** Data-field defaults for useStudioStore (actions attached in studio.ts). */
export const studioInitialData = {
  sourceMode: "model",
  equations: ["dv/dt = -(v - E_L) / tau_m + I / C"],
  threshold: "v > -50", reset: "v = -65",
  odeParams: { E_L: -65, tau_m: 10, C: 1 },
  odeInit: { v: -65 },
  models: [], selectedModelName: "", modelDetail: null, modelParams: {},
  capabilities: [], capabilitiesLoading: false, capabilitiesError: null,
  authSession: null, authLoading: false, authError: null,
  auditStatus: null, auditExport: null,
  auditArchive: null, auditArchiveRetention: null, auditArchivePurge: null,
  auditArchiveRestore: null, auditArchiveValidation: null,
  evidenceBundle: null, evidenceBundleError: null, evidenceBundleLoading: false,
  projectEvidenceBundle: null, projectEvidenceBundleError: null, projectEvidenceBundleLoading: false,
  compileEvidenceBundle: null, compileEvidenceBundleError: null, compileEvidenceBundleLoading: false,
  synthesisEvidenceBundle: null, synthesisEvidenceBundleError: null, synthesisEvidenceBundleLoading: false,
  jobStatus: null, jobRecords: [],
  identityBrowserUsers: [], identityServiceAccounts: [], operatorStatus: null,
  auditLoading: false, auditError: null,
  templates: [], presets: [],
  dt: 0.1, duration: 100, current: 10, protocol: "constant",
  result: null, fiResult: null, bifResult: null, sensResult: null, precResult: null,
  heatmapResult: null, compareResult: null, nullclineResult: null,
  freqResult: null, staResult: null,
  charResult: null, multiResults: null, importedTrace: null, networkResult: null,
  networkParams: { n_exc: 80, n_inh: 20, w_ee: 0.1, w_ei: 0.4, w_ie: 0.1, w_ii: 0.4, p_conn: 0.2, ext_rate: 5.0 },
  verilogSrc: "", irText: "", svSource: "", irErrors: [] as string[], compileTraceability: null,
  progressPct: 0, progressMsg: "",
  graphPopulations: [], graphProjections: [], graphModels: [], graphSimResult: null, graphErrors: [],
  projectSaveResult: null, serverProjects: [], pipelineResult: null,
  synthTarget: "ice40", synthResult: null, synthEstimate: null, multiTargetResult: null,
  latestSynthesisJobId: null, latestMultiTargetSynthesisJobId: null, toolsAvailable: null,
  trainingJobId: null, trainingStatus: "idle", trainingEpochs: [],
  trainingWeightRestorePlan: null, trainingWeightRestoreVerification: null,
  trainingWeightMaterialization: null, trainingWeightAttach: null,
  trainingWeightLiveAttach: null,
  trainingSurrogates: [],
  trainingConfig: {
    dataset: "synthetic", epochs: 10, batch_size: 64, lr: 0.001,
    hidden: [128], timesteps: 25, surrogate: "atan_surrogate",
    learn_beta: false, learn_threshold: false,
  },
  codeScript: "", codeOneliner: "",
  savedSessions: readStoredStudioSessions(),
  error: null, isSimulating: false,
  activeTab: "trace", modelFilter: "", sweepParam: "", sweepParamY: "",

};
