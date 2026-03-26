import { create } from "zustand";
import {
  fetchTemplates,
  simulate,
  type NeuronTemplate,
  type SimulateResponse,
} from "../api/client";

interface StudioState {
  equations: string[];
  threshold: string;
  reset: string;
  params: Record<string, number>;
  init: Record<string, number>;
  dt: number;
  duration: number;
  current: number;

  templates: NeuronTemplate[];
  selectedTemplate: string;
  result: SimulateResponse | null;
  error: string | null;
  isSimulating: boolean;

  setEquations: (eqs: string[]) => void;
  setThreshold: (t: string) => void;
  setReset: (r: string) => void;
  setParam: (key: string, value: number) => void;
  setInit: (key: string, value: number) => void;
  setDt: (dt: number) => void;
  setDuration: (d: number) => void;
  setCurrent: (c: number) => void;
  loadTemplates: () => Promise<void>;
  selectTemplate: (name: string) => void;
  runSimulation: () => Promise<void>;
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

  templates: [],
  selectedTemplate: "lif",
  result: null,
  error: null,
  isSimulating: false,

  setEquations: (eqs) => set({ equations: eqs }),
  setThreshold: (t) => set({ threshold: t }),
  setReset: (r) => set({ reset: r }),
  setParam: (key, value) =>
    set((s) => ({ params: { ...s.params, [key]: value } })),
  setInit: (key, value) =>
    set((s) => ({ init: { ...s.init, [key]: value } })),
  setDt: (dt) => set({ dt }),
  setDuration: (d) => set({ duration: d }),
  setCurrent: (c) => set({ current: c }),

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
      error: null,
    });
  },

  runSimulation: async () => {
    const s = get();
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
      });
      set({ result, isSimulating: false });
    } catch (e) {
      set({
        error: e instanceof Error ? e.message : String(e),
        isSimulating: false,
      });
    }
  },
}));
