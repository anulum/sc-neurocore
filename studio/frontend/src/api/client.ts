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
  n_state_vars: number; n_params: number; state_var_names: string[];
  dt: number; description: string;
}

export interface ModelDetail {
  name: string; module: string; category: string;
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

export interface CapabilityRequirement {
  name: string;
  available: boolean;
  detail: string;
}

export interface StudioCapability {
  capability_id: string;
  title: string;
  summary: string;
  status: "stable" | "experimental" | "degraded" | "unavailable";
  healthy: boolean;
  message: string;
  requirements: CapabilityRequirement[];
  evidence: string[];
  ui_placement: string;
  docs_path: string | null;
}

export interface StudioCapabilitiesResponse {
  capabilities: StudioCapability[];
}

export interface StudioAuditStatus {
  configured: boolean;
  healthy: boolean;
  last_error: string | null;
  path_configured: boolean;
  sink_type: string;
}

export interface StudioAuditEvent {
  action: string;
  decision: string;
  principal_id: string | null;
  reason: string;
  request_id: string | null;
  route: string;
  schema_version: string;
  timestamp_utc: string | null;
  previous_event_hash: string | null;
  event_hash: string | null;
}

export interface StudioAuditExport {
  configured: boolean;
  event_count: number;
  events: StudioAuditEvent[];
  schema_version: string;
  sink_type: string;
  truncated: boolean;
}

export interface StudioJobStatus {
  active_count: number;
  allowed_kinds: string[];
  completed_count: number;
  configured: boolean;
  failed_count: number;
  schema_version: string;
  timed_out_count: number;
}

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
export const fetchStudioCapabilities = () =>
  get<StudioCapabilitiesResponse>("/studio/capabilities");
export const fetchStudioAuditStatus = () =>
  get<StudioAuditStatus>("/studio/audit/status");
export const fetchStudioAuditExport = (limit = 100) =>
  get<StudioAuditExport>(`/studio/audit/export?limit=${encodeURIComponent(limit)}`);
export const fetchStudioJobStatus = () =>
  get<StudioJobStatus>("/studio/jobs/status");

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
export const fetchModelScan = () => get<ModelBehavior[]>("/models/scan");
export const simulateNetwork = (req: Record<string, unknown>) => post<NetworkResult>("/network/ei", req);

export interface NetworkResult {
  spike_times: number[];
  spike_neurons: number[];
  n_exc: number; n_inh: number; n_total: number; n_spikes: number;
  rate_time: number[]; exc_rates: number[]; inh_rates: number[];
  duration: number; dt: number;
  mean_exc_rate: number; mean_inh_rate: number;
}

export interface ModelBehavior {
  name: string; category: string; pattern: string;
  description: string; rate_hz: number; spike_count: number;
}
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

// --- Compiler Inspector (Block 2) ---

export interface IRBuildResponse {
  ir_text: string;
  errors: string[];
  n_ops: number;
  n_inputs: number;
  n_outputs: number;
  graph_name: string;
  params_q88: Record<string, number>;
}

export interface IRVerifyResponse {
  valid: boolean;
  errors: string[];
  n_ops: number;
  graph_name: string;
}

export interface SVEmitResponse {
  systemverilog: string;
  graph_name: string;
  chars: number;
}

export interface SVDirectResponse {
  verilog: string;
  ir_repr: string;
  chars: number;
  module_name: string;
}

export const buildIR = (req: Record<string, unknown>) => post<IRBuildResponse>("/ir/build", req);
export const verifyIR = (irText: string) => post<IRVerifyResponse>("/ir/verify", { ir_text: irText });
export const emitSV = (irText: string) => post<SVEmitResponse>("/ir/emit-sv", { ir_text: irText });
export const emitSVDirect = (req: Record<string, unknown>) => post<SVDirectResponse>("/ir/emit-sv-direct", req);
export const fetchCosimDetail = (req: Record<string, unknown>) => post<PrecisionResponse>("/ir/cosim", req);

// --- Synthesis Dashboard (Block 3) ---

export interface SynthToolInfo {
  available: boolean;
  version: string | null;
}

export interface SynthResources {
  luts: number;
  ffs: number;
  brams: number;
  dsps: number;
  cells: number;
  wires: number;
}

export interface SynthCapacity {
  luts: number;
  ffs: number;
  brams: number;
  dsps: number;
}

export interface SynthResult {
  success: boolean;
  error?: string;
  target: string;
  resources: SynthResources;
  capacity: SynthCapacity;
  utilisation: Record<string, number>;
  log_excerpt: string;
}

export interface SynthEstimate {
  target: string;
  estimated: boolean;
  resources: { luts: number; ffs: number; brams: number; dsps: number };
  capacity: SynthCapacity;
  utilisation: Record<string, number>;
}

export interface MultiTargetResult {
  targets: Record<string, SynthResult>;
  supported: string[];
}

export interface PnRResult {
  success: boolean;
  error?: string;
  max_freq_mhz: number | null;
  critical_path: string | null;
  log_excerpt: string;
}

export const fetchSynthTools = () => get<Record<string, SynthToolInfo>>("/synth/tools-status");
export const runSynthesis = (verilog: string, target: string) =>
  post<SynthResult>("/synth/run", { verilog, target });
export const runMultiTargetSynthesis = (verilog: string) =>
  post<MultiTargetResult>("/synth/multi-target", { verilog });
export const fetchSynthEstimate = (irOpCount: number, target: string) =>
  post<SynthEstimate>("/synth/estimate", { ir_op_count: irOpCount, target });
export const runPnR = (jsonPath: string, target: string) =>
  post<PnRResult>("/synth/pnr", { json_path: jsonPath, target });

// --- Training Monitor (Block 4) ---

export interface SurrogateInfo {
  name: string;
  available: boolean;
}

export interface CellTypeInfo {
  name: string;
  available: boolean;
}

export interface TrainingConfig {
  dataset: string;
  epochs: number;
  batch_size: number;
  lr: number;
  hidden: number[];
  timesteps: number;
  surrogate: string;
  learn_beta: boolean;
  learn_threshold: boolean;
  max_grad_norm: number;
}

export interface TrainingEpochMetrics {
  epoch: number;
  train_loss: number;
  train_accuracy: number;
  val_loss: number;
  val_accuracy: number;
  layer_spike_rates: Record<string, number>;
  param_snapshot: Record<string, number>;
}

export interface TrainingJobStatus {
  job_id: string;
  status: string;
  error: string | null;
  final_metrics: Record<string, number> | null;
}

export interface TrainingJobSummary {
  job_id: string;
  status: string;
  config: TrainingConfig;
}

export const fetchSurrogates = () => get<SurrogateInfo[]>("/training/surrogates");
export const fetchCellTypes = () => get<CellTypeInfo[]>("/training/cell-types");
export const startTraining = (config: Partial<TrainingConfig>) =>
  post<{ job_id: string; status: string }>("/training/start", config);
export const stopTraining = (jobId: string) =>
  post<{ job_id: string; status: string }>("/training/stop", { job_id: jobId });
export const fetchTrainingStatus = (jobId: string) =>
  get<TrainingJobStatus>(`/training/status/${jobId}`);
export const fetchTrainingJobs = () => get<TrainingJobSummary[]>("/training/jobs");

// --- Network Canvas (Block 5) ---

export interface PopulationNode {
  id: string;
  type: "population";
  label: string;
  model: string;
  count: number;
  neuron_type: "excitatory" | "inhibitory";
  position: { x: number; y: number };
  params: Record<string, number>;
}

export interface ProjectionEdge {
  id: string;
  source: string;
  target: string;
  weight: number;
  delay: number;
  probability: number;
}

export interface NetworkGraph {
  populations: PopulationNode[];
  projections: ProjectionEdge[];
  duration?: number;
  dt?: number;
}

export interface GraphSimResult {
  success: boolean;
  errors?: string[];
  spike_times?: number[];
  spike_neurons?: number[];
  n_exc?: number;
  n_inh?: number;
  n_total?: number;
  n_spikes?: number;
  rate_time?: number[];
  exc_rates?: number[];
  inh_rates?: number[];
  graph_summary?: { n_populations: number; n_projections: number; n_exc: number; n_inh: number };
}

export interface NIRFormat {
  format: string;
  version: string;
  nodes: Record<string, unknown>;
  edges: unknown[];
}

export const fetchGraphModels = () => get<string[]>("/graph/models");
export const createPopulation = (data: Partial<PopulationNode>) =>
  post<PopulationNode>("/graph/population", data);
export const createProjection = (data: { source_id: string; target_id: string; weight?: number; delay?: number; probability?: number }) =>
  post<ProjectionEdge>("/graph/projection", data);
export const validateGraph = (graph: NetworkGraph) =>
  post<{ valid: boolean; errors: string[] }>("/graph/validate", graph);
export const simulateGraph = (graph: NetworkGraph) =>
  post<GraphSimResult>("/graph/simulate", graph);
export const exportNIR = (graph: NetworkGraph) =>
  post<NIRFormat>("/graph/export-nir", graph);
export const importNIR = (nir: NIRFormat) =>
  post<NetworkGraph>("/graph/import-nir", nir);

// --- Integration (Block 6) ---

export interface ProjectSummary {
  name: string;
  saved_at: number;
  version: string;
}

export interface PipelineResult {
  success: boolean;
  target: string;
  step?: string;
  errors?: string[];
  error?: string;
  steps?: Record<string, unknown>;
  pipeline?: string;
}

export const saveProject = (name: string, state: Record<string, unknown>) =>
  post<{ name: string; path: string; saved_at: number }>("/project/save", { name, state });
export const loadProject = (name: string) => get<Record<string, unknown>>(`/project/load/${name}`);
export const listProjects = () => get<ProjectSummary[]>("/project/list");
export const deleteProject = (name: string) =>
  fetch(`/api/project/${name}`, { method: "DELETE" }).then((r) => json<{ deleted: string }>(r));
export const runPipeline = (graph: NetworkGraph, target: string) =>
  post<PipelineResult>("/pipeline/run", { graph, target });

// --- WebSocket Progress ---

export interface ProgressMessage {
  type: "progress" | "complete" | "error" | "heartbeat";
  step?: string;
  pct?: number;
  msg?: string;
  result?: unknown;
}

export function connectProgress(
  op: string,
  config: Record<string, unknown>,
  onMessage: (msg: ProgressMessage) => void,
): WebSocket {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${window.location.host}/ws/progress`);
  ws.onopen = () => ws.send(JSON.stringify({ op, config }));
  ws.onmessage = (e) => {
    try {
      onMessage(JSON.parse(e.data));
    } catch { /* ignore parse errors */ }
  };
  ws.onerror = () => onMessage({ type: "error", msg: "WebSocket connection failed" });
  return ws;
}
