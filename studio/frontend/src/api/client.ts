export interface SimulateRequest {
  equations: string[];
  threshold?: string | null;
  reset?: string | null;
  params?: Record<string, number> | null;
  init?: Record<string, number> | null;
  dt: number;
  duration: number;
  current: number;
  protocol: string;
}

export interface SpikeStats {
  rate_hz: number;
  isi_mean_ms: number | null;
  isi_cv: number | null;
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

const BASE = "/api";

export async function fetchTemplates(): Promise<NeuronTemplate[]> {
  const r = await fetch(`${BASE}/templates`);
  if (!r.ok) throw new Error(`Templates fetch failed: ${r.status}`);
  return r.json();
}

export async function simulate(req: SimulateRequest): Promise<SimulateResponse> {
  const r = await fetch(`${BASE}/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || `Simulate failed: ${r.status}`);
  }
  return r.json();
}

export async function fetchFICurve(req: {
  equations: string[];
  threshold?: string | null;
  reset?: string | null;
  params?: Record<string, number> | null;
  init?: Record<string, number> | null;
  dt: number;
  duration: number;
  i_min: number;
  i_max: number;
  i_steps: number;
}): Promise<FICurveResponse> {
  const r = await fetch(`${BASE}/fi-curve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || `f-I curve failed: ${r.status}`);
  }
  return r.json();
}
