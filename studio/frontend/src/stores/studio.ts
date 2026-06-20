import { create } from "zustand";
import {
  fetchTemplates, fetchModels, fetchModelDetail, fetchPresets, fetchPreset,
  fetchStudioIdentityServiceAccounts,
  fetchStudioAuditExport, fetchStudioAuditStatus, fetchStudioCapabilities,
  fetchStudioJobs, fetchStudioJobStatus, fetchStudioOperatorStatus,
  simulateODE, simulateModel, fetchFICurve, compileVerilog,
  fetchBifurcation, fetchSensitivity, fetchPrecision, fetchHeatmap, fetchCodegen,
  fetchCompare, fetchNullclines, fetchFreqResponse,
  fetchCharacterize, fetchMultiSimulate, importTrace, simulateNetwork,
  buildIR, emitSV, emitSVDirect,
  fetchSynthTools, runSynthesis as apiRunSynthesis, runMultiTargetSynthesis,
  fetchSynthEstimate,
  fetchSurrogates as apiFetchSurrogates, startTraining as apiStartTraining,
  stopTraining as apiStopTraining,
  fetchGraphModels as apiFetchGraphModels,
  createPopulation as apiCreatePop, createProjection as apiCreateProj,
  simulateGraph as apiSimGraph, validateGraph as apiValidateGraph,
  exportNIR as apiExportNIR, importNIR as apiImportNIR,
  saveProject as apiSaveProject, loadProject as apiLoadProject,
  listProjects as apiListProjects, deleteProject as apiDeleteProject,
  runPipeline as apiRunPipeline,
  type CharacterizeResponse, type ImportedTrace, type NetworkResult,
  type NeuronTemplate, type ModelSummary, type ModelDetail, type PresetSummary,
  type SimulateResponse, type FICurveResponse, type BifurcationResponse,
  type SensitivityResponse, type PrecisionResponse, type HeatmapResponse,
  type CompareResponse, type NullclineResponse, type FreqResponse,
  type SynthResult, type SynthEstimate, type MultiTargetResult,
  type SynthToolInfo,
  type SurrogateInfo, type TrainingEpochMetrics,
  type PopulationNode, type ProjectionEdge, type GraphSimResult, type NIRFormat,
  type ProjectSummary, type PipelineResult,
  type StudioAuditExport, type StudioAuditStatus, type StudioCapability,
  type StudioIdentityServiceAccount, type StudioJobRecord, type StudioJobStatus,
  type StudioOperatorStatus,
  updateStudioIdentityServiceAccount,
  connectProgress,
} from "../api/client";

let debounceTimer: ReturnType<typeof setTimeout> | null = null;

export type SourceMode = "model" | "ode";
export type ViewTab = "trace" | "phase" | "isi" | "fi-curve" | "bifurcation" |
  "sensitivity" | "precision" | "heatmap" | "verilog" | "code" |
  "compare" | "freq" | "sta" | "characterize" | "multi" | "network" | "ir" | "synth" | "train" | "canvas" | "admin";

interface StudioState {
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
  capabilities: StudioCapability[];
  capabilitiesLoading: boolean;
  capabilitiesError: string | null;
  auditStatus: StudioAuditStatus | null;
  auditExport: StudioAuditExport | null;
  jobStatus: StudioJobStatus | null;
  jobRecords: StudioJobRecord[];
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
  networkParams: { n_exc: number; n_inh: number; w_ee: number; w_ei: number; w_ie: number; w_ii: number; p_conn: number; ext_rate: number };
  verilogSrc: string;
  irText: string;
  svSource: string;
  irErrors: string[];
  synthTarget: string;
  synthResult: SynthResult | null;
  synthEstimate: SynthEstimate | null;
  multiTargetResult: MultiTargetResult | null;
  toolsAvailable: Record<string, SynthToolInfo> | null;
  graphPopulations: PopulationNode[];
  graphProjections: ProjectionEdge[];
  graphModels: string[];
  graphSimResult: GraphSimResult | null;
  progressPct: number;
  progressMsg: string;
  graphErrors: string[];
  serverProjects: ProjectSummary[];
  pipelineResult: PipelineResult | null;
  trainingJobId: string | null;
  trainingStatus: string;
  trainingEpochs: TrainingEpochMetrics[];
  trainingSurrogates: SurrogateInfo[];
  trainingConfig: {
    dataset: string; epochs: number; batch_size: number; lr: number;
    hidden: number[]; timesteps: number; surrogate: string;
    learn_beta: boolean; learn_threshold: boolean;
  };
  codeScript: string;
  codeOneliner: string;
  savedSessions: { name: string; state: Record<string, unknown> }[];
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
  loadAuditStatus: () => Promise<void>;
  loadAuditExport: () => Promise<void>;
  loadJobStatus: () => Promise<void>;
  loadIdentityServiceAccounts: () => Promise<void>;
  updateIdentityServiceAccount: (
    principalId: string,
    update: { active: boolean; expires_at_utc: string | null; roles: string[] },
  ) => Promise<void>;
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
  setNetworkParam: (key: string, value: number) => void;
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
  setTrainingConfig: (key: string, value: unknown) => void;
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

function currentConfig(s: StudioState): Record<string, unknown> {
  if (s.sourceMode === "model" && s.selectedModelName) {
    return {
      model_name: s.selectedModelName, params: s.modelParams,
      dt: s.dt, duration: s.duration, current: s.current, protocol: s.protocol,
    };
  }
  return {
    equations: s.equations, threshold: s.threshold || null, reset: s.reset || null,
    params: s.odeParams, init: s.odeInit,
    dt: s.dt, duration: s.duration, current: s.current, protocol: s.protocol,
  };
}

export const useStudioStore = create<StudioState>((set, get) => ({
  sourceMode: "model",
  equations: ["dv/dt = -(v - E_L) / tau_m + I / C"],
  threshold: "v > -50", reset: "v = -65",
  odeParams: { E_L: -65, tau_m: 10, C: 1 },
  odeInit: { v: -65 },
  models: [], selectedModelName: "", modelDetail: null, modelParams: {},
  capabilities: [], capabilitiesLoading: false, capabilitiesError: null,
  auditStatus: null, auditExport: null, jobStatus: null, jobRecords: [],
  identityServiceAccounts: [], operatorStatus: null,
  auditLoading: false, auditError: null,
  templates: [], presets: [],
  dt: 0.1, duration: 100, current: 10, protocol: "constant",
  result: null, fiResult: null, bifResult: null, sensResult: null, precResult: null,
  heatmapResult: null, compareResult: null, nullclineResult: null,
  freqResult: null, staResult: null,
  charResult: null, multiResults: null, importedTrace: null, networkResult: null,
  networkParams: { n_exc: 80, n_inh: 20, w_ee: 0.1, w_ei: 0.4, w_ie: 0.1, w_ii: 0.4, p_conn: 0.2, ext_rate: 5.0 },
  verilogSrc: "", irText: "", svSource: "", irErrors: [] as string[],
  progressPct: 0, progressMsg: "",
  graphPopulations: [], graphProjections: [], graphModels: [], graphSimResult: null, graphErrors: [],
  serverProjects: [], pipelineResult: null,
  synthTarget: "ice40", synthResult: null, synthEstimate: null, multiTargetResult: null, toolsAvailable: null,
  trainingJobId: null, trainingStatus: "idle", trainingEpochs: [], trainingSurrogates: [],
  trainingConfig: {
    dataset: "synthetic", epochs: 10, batch_size: 64, lr: 0.001,
    hidden: [128], timesteps: 25, surrogate: "atan_surrogate",
    learn_beta: false, learn_threshold: false,
  },
  codeScript: "", codeOneliner: "",
  savedSessions: JSON.parse(localStorage.getItem("sc-studio-sessions") || "[]"),
  error: null, isSimulating: false,
  activeTab: "trace", modelFilter: "", sweepParam: "", sweepParamY: "",

  setSourceMode: (m) => set({ sourceMode: m }),
  setEquations: (eqs) => { set({ equations: eqs }); get().autoSimulate(); },
  setThreshold: (t) => { set({ threshold: t }); get().autoSimulate(); },
  setReset: (r) => { set({ reset: r }); get().autoSimulate(); },
  setOdeParam: (key, value) => { set((s) => ({ odeParams: { ...s.odeParams, [key]: value } })); get().autoSimulate(); },
  setOdeInit: (key, value) => { set((s) => ({ odeInit: { ...s.odeInit, [key]: value } })); get().autoSimulate(); },
  setModelParam: (key, value) => { set((s) => ({ modelParams: { ...s.modelParams, [key]: value } })); get().autoSimulate(); },
  setDt: (dt) => { set({ dt }); get().autoSimulate(); },
  setDuration: (d) => { set({ duration: d }); get().autoSimulate(); },
  setCurrent: (c) => { set({ current: c }); get().autoSimulate(); },
  setProtocol: (p) => { set({ protocol: p }); get().autoSimulate(); },
  setActiveTab: (tab) => set({ activeTab: tab }),
  setModelFilter: (f) => set({ modelFilter: f }),
  setSweepParam: (p) => set({ sweepParam: p }),
  setSweepParamY: (p) => set({ sweepParamY: p }),

  loadTemplates: async () => set({ templates: await fetchTemplates() }),
  loadCapabilities: async () => {
    set({ capabilitiesLoading: true, capabilitiesError: null });
    try {
      const response = await fetchStudioCapabilities();
      set({
        capabilities: response.capabilities,
        capabilitiesLoading: false,
        capabilitiesError: null,
      });
    } catch (error: unknown) {
      set({
        capabilitiesLoading: false,
        capabilitiesError: error instanceof Error ? error.message : "Capability check failed",
      });
    }
  },
  loadAuditStatus: async () => {
    set({ auditLoading: true, auditError: null });
    try {
      const auditStatus = await fetchStudioAuditStatus();
      set({ auditStatus, auditLoading: false, auditError: null });
    } catch (error: unknown) {
      set({
        auditLoading: false,
        auditError: error instanceof Error ? error.message : "Audit status check failed",
      });
    }
  },
  loadAuditExport: async () => {
    set({ auditLoading: true, auditError: null });
    try {
      const auditExport = await fetchStudioAuditExport(100);
      set({ auditExport, auditLoading: false, auditError: null });
    } catch (error: unknown) {
      set({
        auditLoading: false,
        auditError: error instanceof Error ? error.message : "Audit export failed",
      });
    }
  },
  loadJobStatus: async () => {
    set({ auditLoading: true, auditError: null });
    try {
      const [jobStatus, jobList] = await Promise.all([
        fetchStudioJobStatus(),
        fetchStudioJobs(),
      ]);
      set({ jobStatus, jobRecords: jobList.jobs, auditLoading: false, auditError: null });
    } catch (error: unknown) {
      set({
        auditLoading: false,
        auditError: error instanceof Error ? error.message : "Job status check failed",
      });
    }
  },
  loadIdentityServiceAccounts: async () => {
    set({ auditLoading: true, auditError: null });
    try {
      const response = await fetchStudioIdentityServiceAccounts();
      set({
        auditLoading: false,
        auditError: null,
        identityServiceAccounts: response.service_accounts,
      });
    } catch (error: unknown) {
      set({
        auditLoading: false,
        auditError: error instanceof Error ? error.message : "Identity account check failed",
      });
    }
  },
  updateIdentityServiceAccount: async (principalId, update) => {
    set({ auditLoading: true, auditError: null });
    try {
      await updateStudioIdentityServiceAccount(principalId, update);
      const [identityResponse, auditExport] = await Promise.all([
        fetchStudioIdentityServiceAccounts(),
        fetchStudioAuditExport(100),
      ]);
      set({
        auditExport,
        auditLoading: false,
        auditError: null,
        identityServiceAccounts: identityResponse.service_accounts,
      });
    } catch (error: unknown) {
      set({
        auditLoading: false,
        auditError: error instanceof Error ? error.message : "Identity account update failed",
      });
    }
  },
  loadOperatorStatus: async () => {
    set({ auditLoading: true, auditError: null });
    try {
      const [operatorStatus, jobList] = await Promise.all([
        fetchStudioOperatorStatus(),
        fetchStudioJobs(),
      ]);
      set({
        auditStatus: operatorStatus.audit,
        jobStatus: operatorStatus.jobs,
        jobRecords: jobList.jobs,
        operatorStatus,
        auditLoading: false,
        auditError: null,
      });
    } catch (error: unknown) {
      set({
        auditLoading: false,
        auditError: error instanceof Error ? error.message : "Operator status check failed",
      });
    }
  },
  loadModels: async () => {
    const models = await fetchModels();
    set({ models });
    if (models.length > 0 && !get().selectedModelName) await get().selectModel(models[0].name);
  },
  loadPresets: async () => set({ presets: await fetchPresets() }),

  selectTemplate: (name) => {
    const t = get().templates.find((t) => t.name === name);
    if (!t) return;
    set({
      sourceMode: "ode", equations: t.equations, threshold: t.threshold, reset: t.reset,
      odeParams: { ...t.params }, odeInit: { ...t.init },
      dt: t.dt, duration: t.duration, current: t.current,
      result: null, fiResult: null, error: null,
    });
    get().runSimulation();
  },

  selectModel: async (name) => {
    set({ selectedModelName: name, result: null, fiResult: null, error: null });
    const detail = await fetchModelDetail(name);
    if (!detail) return;
    const params: Record<string, number> = {};
    for (const p of detail.params) params[p.name] = p.default;
    for (const s of detail.state_vars) params[s.name] = s.default;
    set({ modelDetail: detail, modelParams: params, dt: detail.dt, sourceMode: "model" });
    get().runSimulation();
  },

  loadPreset: async (id) => {
    const preset = await fetchPreset(id);
    if (!preset) return;
    if (preset.mode === "model" && preset.model_name) {
      await get().selectModel(preset.model_name as string);
      set({
        current: (preset.current as number) || 10,
        duration: (preset.duration as number) || 200,
        protocol: (preset.protocol as string) || "constant",
      });
    } else if (preset.equations) {
      set({
        sourceMode: "ode",
        equations: preset.equations as string[],
        threshold: (preset.threshold as string) || "",
        reset: (preset.reset as string) || "",
        odeParams: (preset.params as Record<string, number>) || {},
        odeInit: (preset.init as Record<string, number>) || {},
        dt: (preset.dt as number) || 0.1,
        duration: (preset.duration as number) || 200,
        current: (preset.current as number) || 10,
        protocol: (preset.protocol as string) || "constant",
      });
    }
    const view = preset.suggested_view as string;
    if (view === "fi-curve") { get().runFICurve(); }
    else if (view === "precision") { get().runPrecision(); }
    else {
      set({ activeTab: (view as ViewTab) || "trace" });
      get().runSimulation();
    }
  },

  autoSimulate: () => {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => get().runSimulation(), 250);
  },

  runSimulation: async () => {
    const s = get();
    if (s.isSimulating) return;
    set({ isSimulating: true, error: null });
    try {
      const cfg = currentConfig(s);
      const result = s.sourceMode === "model" && s.selectedModelName
        ? await simulateModel(cfg) : await simulateODE(cfg);
      set({ result, isSimulating: false });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e), isSimulating: false });
    }
  },

  runFICurve: async () => {
    const s = get();
    if (s.isSimulating) return;
    set({ isSimulating: true, error: null, activeTab: "fi-curve" });
    try {
      const cfg = currentConfig(s);
      const fiResult = await fetchFICurve({
        ...cfg, i_min: 0, i_max: Math.abs(s.current) * 2 || 50, i_steps: 25,
      });
      set({ fiResult, isSimulating: false });
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e), isSimulating: false }); }
  },

  runBifurcation: async () => {
    const s = get();
    if (s.isSimulating || !s.sweepParam) return;
    set({ isSimulating: true, error: null, activeTab: "bifurcation" });
    try {
      const cfg = currentConfig(s);
      const paramVal = (s.sourceMode === "model" ? s.modelParams : s.odeParams)[s.sweepParam] ?? 0;
      const bifResult = await fetchBifurcation({
        ...cfg, sweep_param: s.sweepParam,
        sweep_min: paramVal * 0.2, sweep_max: paramVal * 3, sweep_steps: 40,
      });
      set({ bifResult, isSimulating: false });
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e), isSimulating: false }); }
  },

  runSensitivity: async () => {
    const s = get();
    if (s.isSimulating) return;
    set({ isSimulating: true, error: null, activeTab: "sensitivity" });
    try {
      const cfg = currentConfig(s);
      const sensResult = await fetchSensitivity(cfg);
      set({ sensResult, isSimulating: false });
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e), isSimulating: false }); }
  },

  runPrecision: async () => {
    const s = get();
    if (s.sourceMode !== "ode") {
      set({ error: "Precision compare only for custom ODE mode" });
      return;
    }
    set({ isSimulating: true, error: null, activeTab: "precision" });
    try {
      const precResult = await fetchPrecision({
        equations: s.equations, threshold: s.threshold, reset: s.reset,
        params: s.odeParams, init: s.odeInit, dt: s.dt, duration: s.duration, current: s.current,
      });
      set({ precResult, isSimulating: false });
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e), isSimulating: false }); }
  },

  runCompile: async () => {
    const s = get();
    if (s.sourceMode !== "ode") { set({ error: "Verilog compile only for ODE mode" }); return; }
    set({ isSimulating: true, error: null, activeTab: "verilog" });
    try {
      const res = await compileVerilog({
        equations: s.equations, threshold: s.threshold, reset: s.reset, params: s.odeParams,
      });
      set({ verilogSrc: res.verilog, isSimulating: false });
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e), isSimulating: false }); }
  },

  runHeatmap: async () => {
    const s = get();
    if (s.isSimulating || !s.sweepParam || !s.sweepParamY) return;
    set({ isSimulating: true, error: null, activeTab: "heatmap" });
    try {
      const params = s.sourceMode === "model" ? s.modelParams : s.odeParams;
      const xVal = params[s.sweepParam] ?? 0;
      const yVal = params[s.sweepParamY] ?? 0;
      const cfg = currentConfig(s);
      const heatmapResult = await fetchHeatmap({
        ...cfg,
        param_x: s.sweepParam, x_min: xVal * 0.2, x_max: xVal * 3, x_steps: 15,
        param_y: s.sweepParamY, y_min: yVal * 0.2, y_max: yVal * 3, y_steps: 15,
      });
      set({ heatmapResult, isSimulating: false });
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e), isSimulating: false }); }
  },

  runCodegen: async () => {
    const s = get();
    set({ activeTab: "code" });
    try {
      const res = await fetchCodegen({
        mode: s.sourceMode,
        model_name: s.sourceMode === "model" ? s.selectedModelName : null,
        equations: s.sourceMode === "ode" ? s.equations : null,
        threshold: s.threshold, reset: s.reset,
        params: s.sourceMode === "model" ? s.modelParams : s.odeParams,
        init: s.sourceMode === "ode" ? s.odeInit : null,
        dt: s.dt, duration: s.duration, current: s.current,
      });
      set({ codeScript: res.script, codeOneliner: res.oneliner });
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e) }); }
  },

  exportData: () => {
    const { result } = get();
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `simulation_${result.model_name || "custom"}.json`;
    a.click();
  },

  exportCSV: () => {
    const { result } = get();
    if (!result) return;
    const vars = Object.keys(result.states);
    const header = ["time", ...vars, "current"].join(",");
    const rows = result.time.map((t, i) => {
      const vals = vars.map((v) => result.states[v][i]?.toFixed(6) ?? "");
      return [t.toFixed(4), ...vals, result.current_trace[i]?.toFixed(4) ?? ""].join(",");
    });
    const csv = [header, ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `simulation_${result.model_name || "custom"}.csv`;
    a.click();
  },

  exportSVG: () => {
    const { result } = get();
    if (!result) {
      const canvas = document.querySelector("canvas");
      if (!canvas) return;
      const a = document.createElement("a");
      a.href = canvas.toDataURL("image/png", 1.0);
      a.download = "sc_neurocore_plot.png";
      a.click();
      return;
    }
    const w = 800, h = 400;
    const pad = { top: 20, right: 20, bottom: 40, left: 60 };
    const pw = w - pad.left - pad.right, ph = h - pad.top - pad.bottom;
    const vars = Object.keys(result.states);
    const colors = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373", "#ce93d8"];
    const allY = vars.flatMap((v) => result.states[v]);
    const yMin = Math.min(...allY), yMax = Math.max(...allY);
    const yRange = yMax - yMin || 1;
    const xMin = result.time[0], xMax = result.time[result.time.length - 1];
    const xRange = xMax - xMin || 1;
    const toX = (t: number) => pad.left + ((t - xMin) / xRange) * pw;
    const toY = (v: number) => pad.top + (1 - (v - yMin) / yRange) * ph;
    let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">\n`;
    svg += `<rect width="${w}" height="${h}" fill="#0d1117"/>\n`;
    for (let i = 0; i <= 4; i++) { const y = pad.top + (ph * i) / 4; svg += `<line x1="${pad.left}" y1="${y}" x2="${pad.left + pw}" y2="${y}" stroke="#1a1f2a" stroke-width="0.5"/>\n`; }
    const stride = Math.max(1, Math.floor(result.time.length / 2000));
    for (let vi = 0; vi < vars.length; vi++) {
      const values = result.states[vars[vi]];
      const pts: string[] = [];
      for (let i = 0; i < result.time.length; i += stride) pts.push(`${toX(result.time[i]).toFixed(1)},${toY(values[i]).toFixed(1)}`);
      svg += `<polyline points="${pts.join(" ")}" fill="none" stroke="${colors[vi % colors.length]}" stroke-width="1.5"/>\n`;
    }
    for (const idx of result.spikes.slice(0, 200)) { const x = toX(result.time[idx] ?? idx * result.dt); svg += `<line x1="${x.toFixed(1)}" y1="${pad.top}" x2="${x.toFixed(1)}" y2="${pad.top + 8}" stroke="#ff5252" stroke-width="1.5"/>\n`; }
    svg += `<line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${pad.top + ph}" stroke="#484f58"/>\n`;
    svg += `<line x1="${pad.left}" y1="${pad.top + ph}" x2="${pad.left + pw}" y2="${pad.top + ph}" stroke="#484f58"/>\n`;
    svg += `<text x="${pad.left + pw / 2}" y="${h - 5}" text-anchor="middle" fill="#8b949e" font-size="11" font-family="sans-serif">time (ms)</text>\n`;
    svg += `<text x="12" y="${pad.top + ph / 2}" text-anchor="middle" fill="#8b949e" font-size="11" font-family="sans-serif" transform="rotate(-90,12,${pad.top + ph / 2})">mV</text>\n`;
    for (let i = 0; i <= 4; i++) { const val = yMin + (yRange * i) / 4; svg += `<text x="${pad.left - 5}" y="${toY(val) + 3}" text-anchor="end" fill="#8b949e" font-size="9" font-family="monospace">${val.toFixed(1)}</text>\n`; }
    for (let vi = 0; vi < vars.length; vi++) { svg += `<line x1="${pad.left + vi * 80}" y1="10" x2="${pad.left + vi * 80 + 15}" y2="10" stroke="${colors[vi % colors.length]}" stroke-width="2"/><text x="${pad.left + vi * 80 + 18}" y="13" fill="#8b949e" font-size="10">${vars[vi]}</text>\n`; }
    if (result.model_name) svg += `<text x="${w - pad.right}" y="13" text-anchor="end" fill="#484f58" font-size="9" font-family="monospace">${result.model_name}</text>\n`;
    svg += `</svg>`;
    const blob = new Blob([svg], { type: "image/svg+xml" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `sc_neurocore_${result.model_name || "custom"}.svg`;
    a.click();
  },

  runCharacterize: async () => {
    const s = get();
    if (s.isSimulating || !s.selectedModelName) return;
    set({ isSimulating: true, error: null, activeTab: "characterize", progressPct: 0, progressMsg: "Starting characterisation..." });
    const config = {
      name: s.selectedModelName, params: s.modelParams,
      dt: s.dt, duration: s.duration, current: s.current,
    };
    const ws = connectProgress("characterize", config, (msg) => {
      if (msg.type === "progress") {
        set({ progressPct: msg.pct || 0, progressMsg: msg.msg || "" });
      } else if (msg.type === "complete") {
        set({ charResult: msg.result as CharacterizeResponse, isSimulating: false, progressPct: 100, progressMsg: "" });
      } else if (msg.type === "error") {
        set({ error: msg.msg || "Characterisation failed", isSimulating: false, progressPct: 0, progressMsg: "" });
      }
    });
    ws.onerror = () => {
      // Fallback to HTTP if WebSocket unavailable
      fetchCharacterize(config).then(
        (charResult) => set({ charResult, isSimulating: false, progressPct: 0, progressMsg: "" }),
        (e) => set({ error: e instanceof Error ? e.message : String(e), isSimulating: false, progressPct: 0, progressMsg: "" }),
      );
    };
  },

  runMultiSimulate: async (modelNames) => {
    const s = get();
    if (s.isSimulating) return;
    set({ isSimulating: true, error: null, activeTab: "multi" });
    try {
      const configs = modelNames.slice(0, 4).map((name) => ({
        name, params: null, dt: null, duration: s.duration, current: s.current, protocol: s.protocol,
      }));
      const multiResults = await fetchMultiSimulate(configs);
      set({ multiResults, isSimulating: false });
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e), isSimulating: false }); }
  },

  setNetworkParam: (key, value) => {
    set((s) => ({ networkParams: { ...s.networkParams, [key]: value } }));
  },

  runNetwork: async () => {
    const s = get();
    if (s.isSimulating) return;
    set({ isSimulating: true, error: null, activeTab: "network" });
    try {
      const np = s.networkParams;
      const networkResult = await simulateNetwork({
        ...np, duration: s.duration,
      });
      set({ networkResult, isSimulating: false });
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e), isSimulating: false }); }
  },

  importCSV: async (csv) => {
    const lines = csv.trim().split("\n").map((l) => l.trim()).filter((l) => l);
    const values: number[] = [];
    for (const line of lines) {
      const parts = line.split(/[,\t\s]+/);
      const num = parseFloat(parts[parts.length - 1]);
      if (!isNaN(num)) values.push(num);
    }
    if (values.length < 10) { set({ error: "Need at least 10 data points" }); return; }
    try {
      const importedTrace = await importTrace({ voltage: values, dt: get().dt });
      set({ importedTrace, activeTab: "trace" });
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e) }); }
  },

  runCompare: async (configB) => {
    const s = get();
    if (s.isSimulating) return;
    set({ isSimulating: true, error: null, activeTab: "compare" });
    try {
      const configA = currentConfig(s);
      const compareResult = await fetchCompare(configA, configB);
      set({ compareResult, isSimulating: false });
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e), isSimulating: false }); }
  },

  runNullclines: async () => {
    const s = get();
    if (s.sourceMode !== "ode" || s.equations.length < 2) {
      set({ error: "Nullclines need 2+ variable ODE in custom mode" });
      return;
    }
    set({ isSimulating: true, error: null });
    try {
      const vars = Object.keys(s.odeInit);
      if (vars.length < 2) { set({ isSimulating: false }); return; }
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
      set({ nullclineResult, isSimulating: false, activeTab: "phase" });
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e), isSimulating: false }); }
  },

  runFreqResponse: async () => {
    const s = get();
    if (s.isSimulating) return;
    set({ isSimulating: true, error: null, activeTab: "freq" });
    try {
      const cfg = currentConfig(s);
      const freqResult = await fetchFreqResponse({
        ...cfg, amplitude: Math.abs(s.current) || 10, freq_min: 1, freq_max: 200, n_freqs: 20,
      });
      set({ freqResult, isSimulating: false });
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e), isSimulating: false }); }
  },

  computeSTA: () => {
    const { result } = get();
    if (!result || result.spikes.length < 3) return;
    const vars = Object.keys(result.states);
    const voltage = result.states[vars[0]];
    const halfWin = Math.min(Math.floor(10 / result.dt), 200);
    const snippets: number[][] = [];
    for (const idx of result.spikes) {
      if (idx - halfWin >= 0 && idx + halfWin < voltage.length) {
        snippets.push(voltage.slice(idx - halfWin, idx + halfWin));
      }
    }
    if (snippets.length === 0) return;
    const avg = snippets[0].map((_, i) =>
      snippets.reduce((sum, s) => sum + s[i], 0) / snippets.length
    );
    const time_ms = avg.map((_, i) => (i - halfWin) * result.dt);
    set({ staResult: { time_ms, average: avg, n_spikes: snippets.length }, activeTab: "sta" });
  },

  runBuildIR: async () => {
    const s = get();
    if (s.sourceMode !== "ode") { set({ error: "IR build requires ODE mode" }); return; }
    set({ isSimulating: true, error: null, activeTab: "ir" });
    try {
      const cfg = {
        equations: s.equations, threshold: s.threshold || null, reset: s.reset || null,
        params: s.odeParams, dt: s.dt,
      };
      const result = await buildIR(cfg);
      set({ irText: result.ir_text, irErrors: result.errors, isSimulating: false });
      if (result.errors.length === 0) {
        const sv = await emitSV(result.ir_text);
        set({ svSource: sv.systemverilog });
      }
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e), isSimulating: false }); }
  },

  runEmitSV: async () => {
    const s = get();
    if (s.sourceMode !== "ode") { set({ error: "SV emit requires ODE mode" }); return; }
    set({ isSimulating: true, error: null, activeTab: "ir" });
    try {
      const result = await emitSVDirect({
        equations: s.equations, threshold: s.threshold || null, reset: s.reset || null,
        params: s.odeParams,
      });
      set({ svSource: result.verilog, irText: result.ir_repr, isSimulating: false });
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e), isSimulating: false }); }
  },

  setSynthTarget: (t) => set({ synthTarget: t }),

  runSynthesis: async () => {
    const s = get();
    if (!s.svSource && !s.verilogSrc) { set({ error: "Generate Verilog first" }); return; }
    set({ isSimulating: true, error: null, activeTab: "synth" });
    try {
      const verilog = s.svSource || s.verilogSrc;
      const synthResult = await apiRunSynthesis(verilog, s.synthTarget);
      set({ synthResult, isSimulating: false });
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e), isSimulating: false }); }
  },

  runMultiTargetSynthesis: async () => {
    const s = get();
    if (!s.svSource && !s.verilogSrc) { set({ error: "Generate Verilog first" }); return; }
    set({ isSimulating: true, error: null, activeTab: "synth" });
    try {
      const verilog = s.svSource || s.verilogSrc;
      const multiTargetResult = await runMultiTargetSynthesis(verilog);
      set({ multiTargetResult, isSimulating: false });
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e), isSimulating: false }); }
  },

  runSynthEstimate: async () => {
    const s = get();
    const irOps = s.irText ? s.irText.split("\n").filter((l) => l.trim().startsWith("%")).length : 0;
    if (irOps < 1) { set({ error: "Build IR first to estimate resources" }); return; }
    try {
      const synthEstimate = await fetchSynthEstimate(irOps, s.synthTarget);
      set({ synthEstimate });
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e) }); }
  },

  checkSynthTools: async () => {
    try {
      const toolsAvailable = await fetchSynthTools();
      set({ toolsAvailable });
    } catch { /* tools check is non-critical */ }
  },

  saveProjectToServer: async (name) => {
    const s = get();
    const state = {
      sourceMode: s.sourceMode, equations: s.equations, threshold: s.threshold,
      reset: s.reset, odeParams: s.odeParams, odeInit: s.odeInit,
      selectedModelName: s.selectedModelName, modelParams: s.modelParams,
      dt: s.dt, duration: s.duration, current: s.current, protocol: s.protocol,
      graphPopulations: s.graphPopulations, graphProjections: s.graphProjections,
      synthTarget: s.synthTarget, trainingConfig: s.trainingConfig,
    };
    try {
      await apiSaveProject(name, state);
      await get().listServerProjects();
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e) }); }
  },

  loadProjectFromServer: async (name) => {
    try {
      const data = await apiLoadProject(name);
      const st = (data as Record<string, unknown>).state as Record<string, unknown> || {};
      set({
        sourceMode: (st.sourceMode as "model" | "ode") || "model",
        equations: (st.equations as string[]) || [],
        threshold: (st.threshold as string) || "",
        reset: (st.reset as string) || "",
        odeParams: (st.odeParams as Record<string, number>) || {},
        odeInit: (st.odeInit as Record<string, number>) || {},
        selectedModelName: (st.selectedModelName as string) || "",
        modelParams: (st.modelParams as Record<string, number>) || {},
        dt: (st.dt as number) || 0.1,
        duration: (st.duration as number) || 100,
        current: (st.current as number) || 10,
        protocol: (st.protocol as string) || "constant",
        graphPopulations: (st.graphPopulations as PopulationNode[]) || [],
        graphProjections: (st.graphProjections as ProjectionEdge[]) || [],
        synthTarget: (st.synthTarget as string) || "ice40",
      });
      get().runSimulation();
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e) }); }
  },

  listServerProjects: async () => {
    try {
      const serverProjects = await apiListProjects();
      set({ serverProjects });
    } catch { /* non-critical */ }
  },

  deleteServerProject: async (name) => {
    try {
      await apiDeleteProject(name);
      await get().listServerProjects();
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e) }); }
  },

  runPipelineAction: async () => {
    const s = get();
    if (s.isSimulating || s.graphPopulations.length === 0) return;
    set({ isSimulating: true, error: null, pipelineResult: null });
    try {
      const graph = { populations: s.graphPopulations, projections: s.graphProjections, duration: s.duration, dt: s.dt };
      const pipelineResult = await apiRunPipeline(graph, s.synthTarget);
      set({ pipelineResult, isSimulating: false });
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e), isSimulating: false }); }
  },

  loadGraphModels: async () => {
    try {
      const graphModels = await apiFetchGraphModels();
      set({ graphModels });
    } catch { /* non-critical */ }
  },

  addPopulation: async (neuronType) => {
    const s = get();
    const idx = s.graphPopulations.length;
    const label = neuronType === "excitatory" ? `Exc ${idx}` : `Inh ${idx}`;
    try {
      const pop = await apiCreatePop({
        label, model: "LIFNeuron", count: neuronType === "excitatory" ? 80 : 20,
        neuron_type: neuronType, x: 100 + idx * 200, y: neuronType === "excitatory" ? 100 : 300,
      } as Record<string, unknown>);
      set((prev) => ({ graphPopulations: [...prev.graphPopulations, pop] }));
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e) }); }
  },

  removePopulation: (id) => {
    set((s) => ({
      graphPopulations: s.graphPopulations.filter((p) => p.id !== id),
      graphProjections: s.graphProjections.filter((e) => e.source !== id && e.target !== id),
    }));
  },

  updatePopulation: (id, updates) => {
    set((s) => ({
      graphPopulations: s.graphPopulations.map((p) => p.id === id ? { ...p, ...updates } : p),
    }));
  },

  addProjection: async (sourceId, targetId) => {
    try {
      const proj = await apiCreateProj({ source_id: sourceId, target_id: targetId, weight: 0.1, probability: 0.2 });
      set((prev) => ({ graphProjections: [...prev.graphProjections, proj] }));
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e) }); }
  },

  removeProjection: (id) => {
    set((s) => ({ graphProjections: s.graphProjections.filter((e) => e.id !== id) }));
  },

  updateProjection: (id, updates) => {
    set((s) => ({
      graphProjections: s.graphProjections.map((e) => e.id === id ? { ...e, ...updates } : e),
    }));
  },

  simulateGraphAction: async () => {
    const s = get();
    if (s.isSimulating) return;
    set({ isSimulating: true, error: null, graphErrors: [] });
    try {
      const graph = { populations: s.graphPopulations, projections: s.graphProjections, duration: s.duration, dt: s.dt };
      const validation = await apiValidateGraph(graph);
      if (!validation.valid) {
        set({ graphErrors: validation.errors, isSimulating: false });
        return;
      }
      const graphSimResult = await apiSimGraph(graph);
      set({ graphSimResult, isSimulating: false });
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e), isSimulating: false }); }
  },

  exportGraphNIR: async () => {
    const s = get();
    try {
      const nir = await apiExportNIR({ populations: s.graphPopulations, projections: s.graphProjections });
      const blob = new Blob([JSON.stringify(nir, null, 2)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "network.nir.json";
      a.click();
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e) }); }
  },

  importGraphNIR: async (nir) => {
    try {
      const graph = await apiImportNIR(nir);
      set({ graphPopulations: graph.populations, graphProjections: graph.projections, activeTab: "canvas" });
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e) }); }
  },

  loadSurrogates: async () => {
    try {
      const trainingSurrogates = await apiFetchSurrogates();
      set({ trainingSurrogates });
    } catch { /* non-critical */ }
  },

  startTraining: async () => {
    const s = get();
    if (s.trainingStatus === "running") return;
    set({ trainingStatus: "starting", trainingEpochs: [], error: null, activeTab: "train" });
    try {
      const result = await apiStartTraining(s.trainingConfig);
      set({ trainingJobId: result.job_id, trainingStatus: "running" });
      // Start SSE listener
      const evtSource = new EventSource(`/api/training/stream/${result.job_id}`);
      evtSource.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (msg.event === "epoch") {
            set((prev) => ({ trainingEpochs: [...prev.trainingEpochs, msg.data] }));
          } else if (msg.event === "completed" || msg.event === "stopped") {
            set({ trainingStatus: msg.event });
            evtSource.close();
          } else if (msg.event === "error") {
            set({ trainingStatus: "failed", error: msg.data.message });
            evtSource.close();
          }
        } catch { /* ignore parse errors */ }
      };
      evtSource.onerror = () => {
        set({ trainingStatus: "disconnected" });
        evtSource.close();
      };
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e), trainingStatus: "failed" });
    }
  },

  stopTraining: async () => {
    const s = get();
    if (!s.trainingJobId) return;
    try {
      await apiStopTraining(s.trainingJobId);
      set({ trainingStatus: "stopping" });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },

  setTrainingConfig: (key, value) => {
    set((s) => ({ trainingConfig: { ...s.trainingConfig, [key]: value } }));
  },

  resetDefaults: () => {
    const s = get();
    if (s.sourceMode === "model" && s.modelDetail) {
      const params: Record<string, number> = {};
      for (const p of s.modelDetail.params) params[p.name] = p.default;
      for (const sv of s.modelDetail.state_vars) params[sv.name] = sv.default;
      set({ modelParams: params, dt: s.modelDetail.dt, current: 10, duration: 100 });
    }
    get().runSimulation();
  },

  saveSession: (name) => {
    const s = get();
    const state = {
      sourceMode: s.sourceMode, equations: s.equations, threshold: s.threshold,
      reset: s.reset, odeParams: s.odeParams, odeInit: s.odeInit,
      selectedModelName: s.selectedModelName, modelParams: s.modelParams,
      dt: s.dt, duration: s.duration, current: s.current, protocol: s.protocol,
    };
    const sessions = s.savedSessions.filter((ss) => ss.name !== name);
    sessions.unshift({ name, state });
    set({ savedSessions: sessions });
    localStorage.setItem("sc-studio-sessions", JSON.stringify(sessions));
  },

  loadSession: (name) => {
    const session = get().savedSessions.find((ss) => ss.name === name);
    if (!session) return;
    const st = session.state as Record<string, unknown>;
    set({
      sourceMode: (st.sourceMode as SourceMode) || "model",
      equations: (st.equations as string[]) || [],
      threshold: (st.threshold as string) || "",
      reset: (st.reset as string) || "",
      odeParams: (st.odeParams as Record<string, number>) || {},
      odeInit: (st.odeInit as Record<string, number>) || {},
      selectedModelName: (st.selectedModelName as string) || "",
      modelParams: (st.modelParams as Record<string, number>) || {},
      dt: (st.dt as number) || 0.1,
      duration: (st.duration as number) || 100,
      current: (st.current as number) || 10,
      protocol: (st.protocol as string) || "constant",
    });
    get().runSimulation();
  },

  deleteSession: (name) => {
    const sessions = get().savedSessions.filter((ss) => ss.name !== name);
    set({ savedSessions: sessions });
    localStorage.setItem("sc-studio-sessions", JSON.stringify(sessions));
  },

  shareURL: () => {
    const s = get();
    const state = {
      m: s.sourceMode, mn: s.selectedModelName, eq: s.equations,
      th: s.threshold, rs: s.reset,
      p: s.sourceMode === "model" ? s.modelParams : s.odeParams,
      i: s.odeInit, dt: s.dt, d: s.duration, c: s.current, pr: s.protocol,
    };
    const encoded = btoa(JSON.stringify(state));
    const url = `${window.location.origin}${window.location.pathname}#${encoded}`;
    navigator.clipboard.writeText(url);
    set({ error: "URL copied to clipboard" });
    setTimeout(() => set({ error: null }), 2000);
  },
}));

// Load from URL hash on startup
try {
  const hash = window.location.hash.slice(1);
  if (hash) {
    const state = JSON.parse(atob(hash));
    if (state.m && state.mn) {
      useStudioStore.getState().selectModel(state.mn);
      useStudioStore.setState({
        current: state.c || 10, duration: state.d || 100,
        protocol: state.pr || "constant",
      });
    }
  }
} catch { /* ignore invalid hash */ }
