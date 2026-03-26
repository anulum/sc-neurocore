import { create } from "zustand";
import {
  fetchTemplates,
  simulate,
  fetchFICurve,
  type NeuronTemplate,
  type SimulateResponse,
  type FICurveResponse,
} from "../api/client";

let debounceTimer: ReturnType<typeof setTimeout> | null = null;

type ViewTab = "trace" | "fi-curve";

interface StudioState {
  equations: string[];
  threshold: string;
  reset: string;
  params: Record<string, number>;
  init: Record<string, number>;
  dt: number;
  duration: number;
  current: number;
  protocol: string;

  templates: NeuronTemplate[];
  selectedTemplate: string;
  result: SimulateResponse | null;
  fiResult: FICurveResponse | null;
  error: string | null;
  isSimulating: boolean;
  activeTab: ViewTab;

  setEquations: (eqs: string[]) => void;
  setThreshold: (t: string) => void;
  setReset: (r: string) => void;
  setParam: (key: string, value: number) => void;
  setInit: (key: string, value: number) => void;
  setDt: (dt: number) => void;
  setDuration: (d: number) => void;
  setCurrent: (c: number) => void;
  setProtocol: (p: string) => void;
  setActiveTab: (tab: ViewTab) => void;
  loadTemplates: () => Promise<void>;
  selectTemplate: (name: string) => void;
  runSimulation: () => Promise<void>;
  runFICurve: () => Promise<void>;
  autoSimulate: () => void;
  exportData: () => void;
}

export const useStudioStore = create<StudioState>((set, get) => ({
  equations: ["dv/dt = -(v - E_L) / tau_m + I / C"],
  threshold: "v > -50",
  reset: "v = -65",
  params: { E_L: -65, tau_m: 10, C: 1 },
  init: { v: -65 },
  dt: 0.1,
  duration: 100,
  current: 30,
  protocol: "constant",

  templates: [],
  selectedTemplate: "lif",
  result: null,
  fiResult: null,
  error: null,
  isSimulating: false,
  activeTab: "trace",

  setEquations: (eqs) => { set({ equations: eqs }); get().autoSimulate(); },
  setThreshold: (t) => { set({ threshold: t }); get().autoSimulate(); },
  setReset: (r) => { set({ reset: r }); get().autoSimulate(); },
  setParam: (key, value) => {
    set((s) => ({ params: { ...s.params, [key]: value } }));
    get().autoSimulate();
  },
  setInit: (key, value) => {
    set((s) => ({ init: { ...s.init, [key]: value } }));
    get().autoSimulate();
  },
  setDt: (dt) => { set({ dt }); get().autoSimulate(); },
  setDuration: (d) => { set({ duration: d }); get().autoSimulate(); },
  setCurrent: (c) => { set({ current: c }); get().autoSimulate(); },
  setProtocol: (p) => { set({ protocol: p }); get().autoSimulate(); },
  setActiveTab: (tab) => set({ activeTab: tab }),

  loadTemplates: async () => {
    const templates = await fetchTemplates();
    set({ templates });
  },

  selectTemplate: (name) => {
    const t = get().templates.find((t) => t.name === name);
    if (!t) return;
    set({
      selectedTemplate: name,
      equations: t.equations,
      threshold: t.threshold,
      reset: t.reset,
      params: { ...t.params },
      init: { ...t.init },
      dt: t.dt,
      duration: t.duration,
      current: t.current,
      result: null,
      fiResult: null,
      error: null,
    });
    get().runSimulation();
  },

  autoSimulate: () => {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      get().runSimulation();
    }, 250);
  },

  runSimulation: async () => {
    const s = get();
    if (s.isSimulating) return;
    set({ isSimulating: true, error: null });
    try {
      const result = await simulate({
        equations: s.equations,
        threshold: s.threshold || null,
        reset: s.reset || null,
        params: s.params,
        init: s.init,
        dt: s.dt,
        duration: s.duration,
        current: s.current,
        protocol: s.protocol,
      });
      set({ result, isSimulating: false });
    } catch (e) {
      set({
        error: e instanceof Error ? e.message : String(e),
        isSimulating: false,
      });
    }
  },

  runFICurve: async () => {
    const s = get();
    if (s.isSimulating) return;
    set({ isSimulating: true, error: null, activeTab: "fi-curve" });
    try {
      const fiResult = await fetchFICurve({
        equations: s.equations,
        threshold: s.threshold || null,
        reset: s.reset || null,
        params: s.params,
        init: s.init,
        dt: s.dt,
        duration: s.duration,
        i_min: 0,
        i_max: Math.abs(s.current) * 2 || 50,
        i_steps: 25,
      });
      set({ fiResult, isSimulating: false });
    } catch (e) {
      set({
        error: e instanceof Error ? e.message : String(e),
        isSimulating: false,
      });
    }
  },

  exportData: () => {
    const { result } = get();
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "simulation.json";
    a.click();
    URL.revokeObjectURL(url);
  },
}));
