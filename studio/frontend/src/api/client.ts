export interface SpikeStats {
  rate_hz: number;
  isi_mean_ms: number | null;
  isi_cv: number | null;
  isi_histogram: { counts: number[]; edges: number[] } | null;
}

export interface FiringPattern {
  pattern: string;
  description: string;
  rate_hz?: number;
  isi_cv?: number;
}

export interface SimulateResponse {
  time: number[];
  states: Record<string, number[]>;
  current_trace: number[];
  spikes: number[];
  spike_count: number;
  stats: SpikeStats;
  pattern?: FiringPattern;
  dt: number;
  n_steps: number;
  model_name?: string;
}

export interface HeatmapResponse {
  param_x: string; x_values: number[];
  param_y: string; y_values: number[];
  rates: number[][];
  rate_min: number; rate_max: number;
}

export interface FICurveResponse { currents: number[]; rates: number[]; }

export interface NeuronTemplate {
  name: string; description: string; equations: string[];
  threshold: string; reset: string; params: Record<string, number>;
  init: Record<string, number>; dt: number; current: number; duration: number;
}

export interface ModelSummary {
  name: string; module: string; category: string;
  n_state_vars: number; n_params: number; state_var_names: string[]; dt: number;
}

export interface ModelDetail {
  name: string; module: string;
  state_vars: { name: string; default: number }[];
  params: { name: string; default: number }[];
  dt: number; docstring: string;
}

export interface PresetSummary {
  id: string; title: string; description: string; suggested_view: string;
}

export interface BifurcationResponse {
  param_name: string; param_values: number[]; attractors: number[][];
}

export interface SensitivityResponse {
  base_rate: number;
  sensitivities: { param: string; sensitivity: number; rate_minus: number; rate_plus: number }[];
}

export interface PrecisionResponse {
  float_result: SimulateResponse;
  fixed_result: SimulateResponse;
  error: { variable: string; max_error: number; mean_error: number; rms_error: number; trace: number[] };
  quantized_params: Record<string, number>;
}

export interface NullclineResponse {
  var_names: string[];
  nullcline_0: { variable: string; points: number[][] };
  nullcline_1: { variable: string; points: number[][] };
}

export interface CompareResponse { a: SimulateResponse; b: SimulateResponse; }

export interface FreqResponse { frequencies_hz: number[]; rates: number[]; amplitude: number; }

const BASE = "/api";

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || `${r.status}`);
  }
  return r.json();
}

function post<T>(path: string, body: unknown): Promise<T> {
  return fetch(`${BASE}${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => json<T>(r));
}

function get<T>(path: string): Promise<T> {
  return fetch(`${BASE}${path}`).then((r) => json<T>(r));
}

export const fetchTemplates = () => get<NeuronTemplate[]>("/templates");
export const fetchModels = () => get<ModelSummary[]>("/models");
export const fetchModelDetail = (name: string) => get<ModelDetail>(`/models/${name}`);
export const fetchPresets = () => get<PresetSummary[]>("/presets");
export const fetchPreset = (id: string) => get<Record<string, unknown>>(`/presets/${id}`);

export const simulateODE = (req: Record<string, unknown>) => post<SimulateResponse>("/simulate", req);
export const simulateModel = (req: Record<string, unknown>) => post<SimulateResponse>("/models/simulate", req);
export const fetchFICurve = (req: Record<string, unknown>) => post<FICurveResponse>("/fi-curve", req);
export const compileVerilog = (req: Record<string, unknown>) => post<{ verilog: string }>("/compile", req);
export const fetchBifurcation = (req: Record<string, unknown>) => post<BifurcationResponse>("/bifurcation", req);
export const fetchSensitivity = (req: Record<string, unknown>) => post<SensitivityResponse>("/sensitivity", req);
export const fetchNullclines = (req: Record<string, unknown>) => post<NullclineResponse>("/nullclines", req);
export const fetchPrecision = (req: Record<string, unknown>) => post<PrecisionResponse>("/precision", req);
export const fetchCompare = (a: Record<string, unknown>, b: Record<string, unknown>) => post<CompareResponse>("/compare", { config_a: a, config_b: b });
export const fetchFreqResponse = (req: Record<string, unknown>) => post<FreqResponse>("/freq-response", req);
export const fetchHeatmap = (req: Record<string, unknown>) => post<HeatmapResponse>("/heatmap", req);
export const fetchCodegen = (req: Record<string, unknown>) => post<{ script: string; oneliner: string }>("/codegen", req);
export const fetchCharacterize = (req: Record<string, unknown>) => post<CharacterizeResponse>("/characterize", req);
export const fetchMultiSimulate = (configs: Record<string, unknown>[]) => post<SimulateResponse[]>("/multi-simulate", configs);
export const importTrace = (data: { voltage: number[]; dt: number }) => post<ImportedTrace>("/import-trace", data);

export interface CharacterizeResponse {
  pattern: { pattern: string; description: string };
  fi_curve: { currents: number[]; rates: number[] };
  threshold_current: number | null;
  max_rate: number;
  state_ranges: Record<string, { min: number; max: number; mean: number }>;
  top_sensitivities: { param: string; rate_change: number }[];
  spike_count: number;
  stats: SpikeStats;
}

export interface ImportedTrace {
  time: number[];
  voltage: number[];
  spikes: number[];
  spike_count: number;
  dt: number;
  stats: { mean: number; std: number; min: number; max: number; threshold_estimate: number };
}
