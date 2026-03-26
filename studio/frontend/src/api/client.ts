export interface SimulateRequest {
  equations: string[];
  threshold?: string | null;
  reset?: string | null;
  params?: Record<string, number> | null;
  init?: Record<string, number> | null;
  dt: number;
  duration: number;
  current: number;
}

export interface SimulateResponse {
  time: number[];
  states: Record<string, number[]>;
  spikes: number[];
  spike_count: number;
  dt: number;
  n_steps: number;
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

export async function simulate(
  req: SimulateRequest
): Promise<SimulateResponse> {
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
