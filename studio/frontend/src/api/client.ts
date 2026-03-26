export interface SpikeStats {
  rate_hz: number;
  isi_mean_ms: number | null;
  isi_cv: number | null;
  isi_histogram: { counts: number[]; edges: number[] } | null;
}

export interface SimulateResponse {
  time: number[];
  states: Record<string, number[]>;
  current_trace: number[];
  spikes: number[];
  spike_count: number;
  stats: SpikeStats;
  dt: number;
  n_steps: number;
  model_name?: string;
}

export interface FICurveResponse {
  currents: number[];
  rates: number[];
}

export interface NeuronTemplate {
  name: string;
  description: string;
  equations: string[];
  threshold: string;
  reset: string;
  params: Record<string, number>;
  init: Record<string, number>;
  dt: number;
  current: number;
  duration: number;
}

export interface ModelSummary {
  name: string;
  module: string;
  category: string;
  n_state_vars: number;
  n_params: number;
  state_var_names: string[];
  dt: number;
}

export interface ModelDetail {
  name: string;
  module: string;
  state_vars: { name: string; default: number }[];
  params: { name: string; default: number }[];
  dt: number;
  docstring: string;
}

const BASE = "/api";

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || `Request failed: ${r.status}`);
  }
  return r.json();
}

export const fetchTemplates = () =>
  fetch(`${BASE}/templates`).then((r) => json<NeuronTemplate[]>(r));

export const fetchModels = () =>
  fetch(`${BASE}/models`).then((r) => json<ModelSummary[]>(r));

export const fetchModelDetail = (name: string) =>
  fetch(`${BASE}/models/${name}`).then((r) => json<ModelDetail>(r));

export const simulateODE = (req: {
  equations: string[];
  threshold?: string | null;
  reset?: string | null;
  params?: Record<string, number> | null;
  init?: Record<string, number> | null;
  dt: number;
  duration: number;
  current: number;
  protocol: string;
}) =>
  fetch(`${BASE}/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  }).then((r) => json<SimulateResponse>(r));

export const simulateModel = (req: {
  name: string;
  params?: Record<string, number> | null;
  dt?: number | null;
  duration: number;
  current: number;
  protocol: string;
}) =>
  fetch(`${BASE}/models/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  }).then((r) => json<SimulateResponse>(r));

export const fetchFICurve = (req: {
  equations?: string[] | null;
  model_name?: string | null;
  threshold?: string | null;
  reset?: string | null;
  params?: Record<string, number> | null;
  dt: number;
  duration: number;
  i_min: number;
  i_max: number;
  i_steps: number;
}) =>
  fetch(`${BASE}/fi-curve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  }).then((r) => json<FICurveResponse>(r));

export const compileVerilog = (req: {
  equations: string[];
  threshold?: string | null;
  reset?: string | null;
  params?: Record<string, number> | null;
  module_name?: string;
}) =>
  fetch(`${BASE}/compile`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  }).then((r) => json<{ verilog: string; module_name: string; chars: number }>(r));
