import { create } from "zustand";
import {
  fetchTemplates, fetchModels, fetchModelDetail, fetchPresets, fetchPreset,
  simulateODE, simulateModel, fetchFICurve, compileVerilog,
  fetchBifurcation, fetchSensitivity, fetchPrecision,
  type NeuronTemplate, type ModelSummary, type ModelDetail, type PresetSummary,
  type SimulateResponse, type FICurveResponse, type BifurcationResponse,
  type SensitivityResponse, type PrecisionResponse,
} from "../api/client";

let debounceTimer: ReturnType<typeof setTimeout> | null = null;

type SourceMode = "model" | "ode";
type ViewTab = "trace" | "phase" | "isi" | "fi-curve" | "bifurcation" |
  "sensitivity" | "precision" | "verilog";

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
  verilogSrc: string;
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
  runCompile: () => Promise<void>;
  autoSimulate: () => void;
  exportData: () => void;
  exportSVG: () => void;
  resetDefaults: () => void;
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
  verilogSrc: "", error: null, isSimulating: false,
  activeTab: "trace", modelFilter: "", sweepParam: "",

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
}));
