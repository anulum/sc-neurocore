import { create } from "zustand";
import {
  fetchTemplates, fetchModels, fetchModelDetail, fetchPresets, fetchPreset,
  simulateODE, simulateModel, fetchFICurve, compileVerilog,
  fetchBifurcation, fetchSensitivity, fetchPrecision, fetchHeatmap, fetchCodegen,
  fetchCompare, fetchNullclines, fetchFreqResponse,
  fetchCharacterize, fetchMultiSimulate, importTrace,
  type CharacterizeResponse, type ImportedTrace,
  type NeuronTemplate, type ModelSummary, type ModelDetail, type PresetSummary,
  type SimulateResponse, type FICurveResponse, type BifurcationResponse,
  type SensitivityResponse, type PrecisionResponse, type HeatmapResponse,
  type CompareResponse, type NullclineResponse, type FreqResponse,
} from "../api/client";

let debounceTimer: ReturnType<typeof setTimeout> | null = null;

type SourceMode = "model" | "ode";
type ViewTab = "trace" | "phase" | "isi" | "fi-curve" | "bifurcation" |
  "sensitivity" | "precision" | "heatmap" | "verilog" | "code" |
  "compare" | "freq" | "sta" | "characterize" | "multi";

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
  verilogSrc: string;
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
  importCSV: (csv: string) => Promise<void>;
  runCompare: (configB: Record<string, unknown>) => Promise<void>;
  runNullclines: () => Promise<void>;
  runFreqResponse: () => Promise<void>;
  computeSTA: () => void;
  autoSimulate: () => void;
  exportData: () => void;
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
  templates: [], presets: [],
  dt: 0.1, duration: 100, current: 10, protocol: "constant",
  result: null, fiResult: null, bifResult: null, sensResult: null, precResult: null,
  heatmapResult: null, compareResult: null, nullclineResult: null,
  freqResult: null, staResult: null,
  charResult: null, multiResults: null, importedTrace: null,
  verilogSrc: "", codeScript: "", codeOneliner: "",
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

  exportSVG: () => {
    const canvas = document.querySelector("canvas");
    if (!canvas) return;
    const a = document.createElement("a");
    a.href = canvas.toDataURL("image/png", 1.0);
    a.download = "sc_neurocore_plot.png";
    a.click();
  },

  runCharacterize: async () => {
    const s = get();
    if (s.isSimulating || !s.selectedModelName) return;
    set({ isSimulating: true, error: null, activeTab: "characterize" });
    try {
      const charResult = await fetchCharacterize({
        name: s.selectedModelName, params: s.modelParams,
        dt: s.dt, duration: s.duration, current: s.current,
      });
      set({ charResult, isSimulating: false });
    } catch (e) { set({ error: e instanceof Error ? e.message : String(e), isSimulating: false }); }
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
