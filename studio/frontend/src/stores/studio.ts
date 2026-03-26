import { create } from "zustand";
import {
  fetchTemplates, fetchModels, fetchModelDetail,
  simulateODE, simulateModel, fetchFICurve, compileVerilog,
  type NeuronTemplate, type ModelSummary, type ModelDetail,
  type SimulateResponse, type FICurveResponse,
} from "../api/client";

let debounceTimer: ReturnType<typeof setTimeout> | null = null;

type SourceMode = "model" | "ode";
type ViewTab = "trace" | "phase" | "isi" | "fi-curve" | "verilog";

interface StudioState {
  sourceMode: SourceMode;
  // ODE mode
  equations: string[];
  threshold: string;
  reset: string;
  odeParams: Record<string, number>;
  odeInit: Record<string, number>;
  // Model mode
  models: ModelSummary[];
  selectedModelName: string;
  modelDetail: ModelDetail | null;
  modelParams: Record<string, number>;
  // Shared
  templates: NeuronTemplate[];
  dt: number;
  duration: number;
  current: number;
  protocol: string;
  result: SimulateResponse | null;
  fiResult: FICurveResponse | null;
  verilogSrc: string;
  error: string | null;
  isSimulating: boolean;
  activeTab: ViewTab;
  modelFilter: string;

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

  loadTemplates: () => Promise<void>;
  loadModels: () => Promise<void>;
  selectTemplate: (name: string) => void;
  selectModel: (name: string) => Promise<void>;
  runSimulation: () => Promise<void>;
  runFICurve: () => Promise<void>;
  runCompile: () => Promise<void>;
  autoSimulate: () => void;
  exportData: () => void;
  resetDefaults: () => void;
}

export const useStudioStore = create<StudioState>((set, get) => ({
  sourceMode: "model",
  equations: ["dv/dt = -(v - E_L) / tau_m + I / C"],
  threshold: "v > -50",
  reset: "v = -65",
  odeParams: { E_L: -65, tau_m: 10, C: 1 },
  odeInit: { v: -65 },
  models: [],
  selectedModelName: "",
  modelDetail: null,
  modelParams: {},
  templates: [],
  dt: 0.1,
  duration: 100,
  current: 10,
  protocol: "constant",
  result: null,
  fiResult: null,
  verilogSrc: "",
  error: null,
  isSimulating: false,
  activeTab: "trace",
  modelFilter: "",

  setSourceMode: (m) => set({ sourceMode: m }),
  setEquations: (eqs) => { set({ equations: eqs }); get().autoSimulate(); },
  setThreshold: (t) => { set({ threshold: t }); get().autoSimulate(); },
  setReset: (r) => { set({ reset: r }); get().autoSimulate(); },
  setOdeParam: (key, value) => {
    set((s) => ({ odeParams: { ...s.odeParams, [key]: value } }));
    get().autoSimulate();
  },
  setOdeInit: (key, value) => {
    set((s) => ({ odeInit: { ...s.odeInit, [key]: value } }));
    get().autoSimulate();
  },
  setModelParam: (key, value) => {
    set((s) => ({ modelParams: { ...s.modelParams, [key]: value } }));
    get().autoSimulate();
  },
  setDt: (dt) => { set({ dt }); get().autoSimulate(); },
  setDuration: (d) => { set({ duration: d }); get().autoSimulate(); },
  setCurrent: (c) => { set({ current: c }); get().autoSimulate(); },
  setProtocol: (p) => { set({ protocol: p }); get().autoSimulate(); },
  setActiveTab: (tab) => set({ activeTab: tab }),
  setModelFilter: (f) => set({ modelFilter: f }),

  loadTemplates: async () => {
    const templates = await fetchTemplates();
    set({ templates });
  },

  loadModels: async () => {
    const models = await fetchModels();
    set({ models });
    if (models.length > 0 && !get().selectedModelName) {
      await get().selectModel(models[0].name);
    }
  },

  selectTemplate: (name) => {
    const t = get().templates.find((t) => t.name === name);
    if (!t) return;
    set({
      sourceMode: "ode",
      equations: t.equations,
      threshold: t.threshold,
      reset: t.reset,
      odeParams: { ...t.params },
      odeInit: { ...t.init },
      dt: t.dt,
      duration: t.duration,
      current: t.current,
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

  autoSimulate: () => {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => get().runSimulation(), 250);
  },

  runSimulation: async () => {
    const s = get();
    if (s.isSimulating) return;
    set({ isSimulating: true, error: null });
    try {
      let result: SimulateResponse;
      if (s.sourceMode === "model" && s.selectedModelName) {
        result = await simulateModel({
          name: s.selectedModelName,
          params: s.modelParams,
          dt: s.dt,
          duration: s.duration,
          current: s.current,
          protocol: s.protocol,
        });
      } else {
        result = await simulateODE({
          equations: s.equations,
          threshold: s.threshold || null,
          reset: s.reset || null,
          params: s.odeParams,
          init: s.odeInit,
          dt: s.dt,
          duration: s.duration,
          current: s.current,
          protocol: s.protocol,
        });
      }
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
      const iMax = Math.abs(s.current) * 2 || 50;
      const fiResult = await fetchFICurve({
        equations: s.sourceMode === "ode" ? s.equations : null,
        model_name: s.sourceMode === "model" ? s.selectedModelName : null,
        threshold: s.sourceMode === "ode" ? s.threshold : null,
        reset: s.sourceMode === "ode" ? s.reset : null,
        params: s.sourceMode === "ode" ? s.odeParams : s.modelParams,
        dt: s.dt,
        duration: s.duration,
        i_min: 0,
        i_max: iMax,
        i_steps: 25,
      });
      set({ fiResult, isSimulating: false });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e), isSimulating: false });
    }
  },

  runCompile: async () => {
    const s = get();
    if (s.sourceMode !== "ode") {
      set({ error: "Verilog compilation only works with custom ODE equations", activeTab: "verilog" });
      return;
    }
    set({ isSimulating: true, error: null, activeTab: "verilog" });
    try {
      const res = await compileVerilog({
        equations: s.equations,
        threshold: s.threshold || null,
        reset: s.reset || null,
        params: s.odeParams,
      });
      set({ verilogSrc: res.verilog, isSimulating: false });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e), isSimulating: false });
    }
  },

  exportData: () => {
    const { result } = get();
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `simulation_${result.model_name || "custom"}.json`;
    a.click();
    URL.revokeObjectURL(url);
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
