// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio frontend API
// Shared Studio API DTO contracts.

/** Spike-train statistics the server computed from a run's own spike times. */
export interface SpikeStats {
  rate_hz: number;
  isi_mean_ms: number | null;
  isi_cv: number | null;
  isi_histogram: { counts: number[]; edges: number[] } | null;
}

/**
 * How a run's spiking was classified, with the sentence shown beside it.
 *
 * The rate and the coefficient of variation are optional because a
 * classification can be made from too few spikes to quote either.
 */
export interface FiringPattern {
  pattern: string;
  description: string;
  rate_hz?: number;
  isi_cv?: number;
}

/**
 * What a run was, so a result can be identified without trusting its shape.
 *
 * `input_sha256` and `result_sha256` are what make a run citable: the same
 * input digest with a different result digest is a reproducibility failure,
 * not a rounding difference. The `v2` fields are optional because a `v1`
 * server does not send them; reading one as absent is correct, reading it as
 * `false` is not.
 */
export interface SimulationRunMetadata {
  dt: number;
  evidence_classification: "simulation";
  input_sha256: string;
  n_steps: number;
  result_sha256: string;
  sample_count: number;
  schema_version: "studio.simulation-run.v1" | "studio.simulation-run.v2";
  source: "ode" | "model";
  spike_count: number;
  status: "completed";
  state_variables: string[];
  /** v2: digest of the full-resolution ``raw`` block. */
  raw_sha256?: string;
  /** v2: whether ``raw`` carries every step. */
  raw_included?: boolean;
  /** v2: ``descriptor``, ``equations`` or ``undeclared``. */
  layout_source?: string;
  /** v2: every declared variable recorded and nothing undeclared changed. */
  state_custody_complete?: boolean;
  /** v2: ``post-step`` sample clock. */
  observation_clock?: string;
}

/**
 * One state variable, and what the server can say about it.
 *
 * `role` and `meaning` come from the model's descriptor where there is one;
 * `observable` and `reason` say whether it can be traced and why not when it
 * cannot, so an absent trace is explained rather than silently missing.
 */
export interface SimulationStateVariable {
  name: string;
  role: "biological" | "auxiliary" | "unassigned";
  unit: string;
  meaning: string;
  declared_init: number | null;
  kind: "scalar" | "vector" | null;
  shape: number[] | null;
  observable: boolean;
  reason: string;
  trace: "per-step" | "snapshots-only" | "none";
}

/**
 * The whole state's custody: what was declared, what was recorded, what was not.
 *
 * `complete` is a claim the server makes and `incomplete_reasons` is what it
 * says when it cannot; `undeclared_mutable` names variables that changed
 * without being declared, which is a defect in the model rather than in the
 * run.
 */
export interface SimulationStateLayout {
  schema_version: "studio.state-layout.v1";
  source: "descriptor" | "equations" | "undeclared";
  schema_profile: string;
  variables: SimulationStateVariable[];
  recorded: string[];
  complete: boolean;
  incomplete_reasons: string[];
  undeclared_mutable: string[];
  custody_notes: string[];
}

/**
 * When each sample was taken relative to the integration step.
 *
 * Stated rather than assumed: a trace read on a different clock than it was
 * written on is off by a step, and the error is invisible in the numbers.
 */
export interface SimulationObservation {
  clock: "post-step";
  dt: number;
  initial_time_ms: number;
  sample_time_ms: string;
  drive_interval_ms: string;
}

/**
 * The full-resolution trace, when the server could afford to send it.
 *
 * `included` and `reason` are the contract: a raw block that was too large is
 * reported as omitted with the budget that omitted it, never as an empty one.
 * `vector_snapshots_only` names variables recorded at snapshots rather than
 * per step, so a caller does not read them as a per-step series.
 */
export interface SimulationRawBlock {
  schema_version: "studio.raw-trace.v1";
  included: boolean;
  element_count: number;
  element_budget: number;
  dt: number;
  n_steps: number;
  sample_time_ms: string;
  drive_interval_ms: string;
  spike_indices: number[];
  spike_times_ms: number[];
  vector_snapshots_only: string[];
  states?: Record<string, number[]>;
  vector_states?: Record<string, number[][]>;
  drive?: number[];
  reason?: string;
}

/**
 * How the display arrays were reduced from the raw trace.
 *
 * `sample_index` maps each display position back to its raw step, and
 * `spikes_are_raw_steps` says which of the two clocks the spike indices are
 * on; without both, a cursor on the plot cannot name the step it is over.
 */
export interface SimulationDisplayProjection {
  schema_version: "studio.display-projection.v1";
  method: "identity" | "bucket-extrema";
  max_points: number;
  bucket_count: number;
  point_count: number;
  sample_index: number[];
  first_sample_included: boolean;
  final_sample_included: boolean;
  spikes_are_raw_steps: boolean;
}

/** One recorded instant: each variable's value, scalar or vector. */
export type SimulationSnapshot = Record<string, number | number[]>;

/** Whether a stochastic run reuses its recorded seed or draws a new one. */
export type StudioTrialMode = "replay" | "fresh";

/**
 * Where a run's randomness came from, in enough detail to reproduce it.
 *
 * `seed_source` distinguishes a seed the caller set from one the model
 * defaulted to and from one the server drew, which is what makes a drawn seed
 * reproducible instead of merely recorded. `effective_trial` is what actually
 * happened, which can differ from what was asked for.
 */
export interface SimulationRandomness {
  kind: "none" | "seeded-model" | "diffusion-noise";
  seed: number | null;
  seed_source: "none" | "request" | "model-default" | "playground-default" | "drawn";
  trial: StudioTrialMode;
  effective_trial: StudioTrialMode;
  generator: string | null;
  note?: string;
}

/**
 * The experiment the server resolved, which is what an export must state.
 *
 * A request is what the caller asked for; this is what was run, with every
 * default filled in and every choice recorded. `experiment_sha256` identifies
 * it, and a generated script checks that digest before reporting a result, so
 * a replay that no longer describes the run says so rather than agreeing by
 * coincidence.
 */
export interface SimulationExperiment {
  schema_version: "studio.experiment-spec.v1";
  source: "model" | "ode";
  model?: Record<string, string>;
  equations?: { equations: string[]; threshold: string | null; reset: string | null; variables: string[]; equation_sha256: string };
  numerical: { method: string; family: string; dt: number; dt_source: string; substeps: number; time_unit: string };
  steps: { n_steps: number; duration_requested_ms: number; duration_effective_ms: number; synchronous_limit: number };
  parameters: Record<string, number | null>;
  initial_state: Record<string, number | null>;
  initial_state_source: string;
  protocol: Record<string, string | number | null>;
  randomness: SimulationRandomness;
  backend: { selected: string; rejected: { name: string; reason: string }[] };
  runtime: Record<string, string>;
  experiment_sha256: string;
  cache: { key: string; cacheable: boolean };
}

/** Whether this result came from the server's cache, and under which key. */
export interface SimulationCacheInfo {
  hit: boolean;
  key: string;
}

/** How much of a metric's domain the server could actually evaluate. */
export type MetricDomain = "complete" | "partial" | "empty";

/** What an analysis computed, in which units, where it is valid and where it could not be evaluated. */
export interface MetricContract {
  schema_version: "studio.metric-contract.v1";
  kind: string;
  definition: string;
  units: Record<string, string>;
  applicability: string[];
  limitations: string[];
  domain: MetricDomain;
  domain_detail: Record<string, unknown>;
}

/**
 * What an analysis was, alongside its result.
 *
 * The same identification an ordinary run carries, plus the contract the
 * payload was computed under and how much of its domain that contract could
 * be evaluated over.
 */
export interface AnalysisResultMetadata {
  analysis_type: string;
  /** Kind of the payload's metric contract, or null for a payload without one. */
  contract?: string | null;
  /** Domain verdict of that contract, or null. */
  domain?: MetricDomain | null;
  evidence_classification: "analysis";
  input_sha256: string;
  output_keys: string[];
  result_sha256: string;
  schema_version: "studio.analysis-result.v1";
  source: "ode" | "model" | "mixed" | "unknown";
  status: "completed";
}

/**
 * One run: the trace to draw, the trace that was computed, and what it was.
 *
 * `time`, `states` and `current_trace` are the display projection, reduced for
 * plotting; `raw` is the full-resolution trace when the server could afford to
 * send it, and `display` says how one maps onto the other. `spikes` is on the
 * raw clock either way, which is why it is never decimated. Everything from
 * `run_metadata` down identifies the run rather than describing its numbers.
 */
export interface SimulateResponse {
  /** Display projection sample times ``(index + 1) * dt``; see ``display``. */
  time: number[];
  /** Display projection of every scalar state; full arrays live in ``raw``. */
  states: Record<string, number[]>;
  /** Display projection of the injected drive. */
  current_trace: number[];
  /** Raw step indices at which the model spiked (never decimated). */
  spikes: number[];
  spike_count: number;
  stats: SpikeStats;
  pattern?: FiringPattern;
  dt: number;
  n_steps: number;
  model_name?: string;
  run_metadata: SimulationRunMetadata;
  observation?: SimulationObservation;
  state_layout?: SimulationStateLayout;
  initial_state?: SimulationSnapshot | null;
  final_state?: SimulationSnapshot;
  raw?: SimulationRawBlock;
  display?: SimulationDisplayProjection;
  experiment?: SimulationExperiment;
  cache?: SimulationCacheInfo;
}

/**
 * A firing rate over a two-parameter sweep, as a grid.
 *
 * `rates` is indexed `[y][x]` to match how it is drawn, and `rate_min` and
 * `rate_max` are the server's own extrema so a colour scale is the same across
 * repeated sweeps.
 */
export interface HeatmapResponse {
  param_x: string; x_values: number[];
  param_y: string; y_values: number[];
  rates: number[][];
  rate_min: number; rate_max: number;
  analysis_metadata: AnalysisResultMetadata;
}

/** Firing rate against injected current, with the contract it was measured under. */
export interface FICurveResponse {
  analysis_metadata: AnalysisResultMetadata;
  contract?: MetricContract;
  currents: number[];
  rates: number[];
}

/**
 * An editable starting point: equations, defaults and a protocol to run them.
 *
 * A template is a text the user may change, not a catalogue model; nothing
 * here is a contract the server will hold a later run to.
 */
export interface NeuronTemplate {
  name: string; description: string; equations: string[];
  threshold: string; reset: string; params: Record<string, number>;
  init: Record<string, number>; dt: number; current: number; duration: number;
}

/**
 * Where a model came from in the literature.
 *
 * `citeable` is the server's judgement that the reference is complete enough
 * to cite, so the Studio does not have to infer it from which fields are
 * filled in.
 */
export interface ModelProvenance {
  authors: string[]; year: number | null; doi: string;
  paper_title: string; url: string; citeable: boolean;
}

/**
 * Metadata health of one catalogue entry.
 *
 * `available` is a declared descriptor, `unavailable` a real model described by
 * code introspection because no descriptor is committed, and `invalid` a model
 * whose metadata could not be read at all.
 */
export type ModelMetadataState = "available" | "unavailable" | "invalid";

/**
 * What the catalogue says about a model without loading its full contract.
 *
 * Enough to browse, filter and judge maturity by; not enough to run. Readiness
 * is stated on two axes because a model can be scientifically well-founded and
 * nowhere near silicon, or the reverse, and one number would hide it.
 */
export interface ModelSummary {
  name: string; module: string; category: string;
  /**
   * Metadata health for this entry. `available` is a declared descriptor,
   * `unavailable` a real model described by code introspection instead, and
   * `invalid` a model whose metadata could not be read. An `invalid` entry is
   * still listed: a catalogue that drops it reports a smaller success count
   * rather than a fault.
   */
  metadata_state: ModelMetadataState;
  /** Why the metadata could not be read, or `null` when it could. */
  metadata_error: string | null;
  tier: number; evidence_kind: string;
  /** Dual-axis science readiness S0–S5 (catalogue contract). */
  science_tier: number;
  science_label: string;
  /** Dual-axis silicon readiness H0–H5, or null when not enrolled. */
  silicon_tier: number | null;
  silicon_label: string;
  /** Descriptor-backed validation, solver, and terminal silicon contract. */
  validation_metric: string;
  integration_method: string;
  terminal_silicon_tier: string;
  terminal_reason: string;
  category_slug: string; category_source: string; family: string;
  maturity: string; biophysical_detail: string;
  n_state_vars: number; n_params: number; state_var_names: string[];
  dt: number; description: string;
  intended_use: string[]; hardware_fit: string[]; behavior_tags: string[];
  provenance: ModelProvenance | null;
}

/**
 * One tunable parameter, with the two ranges that mean different things.
 *
 * `range` is what the model will accept; `biological_range` is what is
 * plausible in a cell. A value inside the first and outside the second is
 * allowed and worth flagging, which is why they are separate.
 */
export interface ModelParameter {
  name: string; default: number; unit: string;
  range: [number, number] | null; biological_range: [number, number] | null;
  meaning: string;
}

/** One state variable's default, unit and meaning, as the model declares it. */
export interface ModelStateVariable {
  name: string; default: number; unit: string; meaning: string;
}

/**
 * Whether one compute backend implements a model, and how closely.
 *
 * `parity` is the interesting field: a backend can be present and not
 * bit-identical, and saying so is more useful than a boolean.
 */
export interface ModelBackendSupport {
  name: string; status: string; parity: string;
}

/**
 * A model's readiness on both axes, with the evidence behind each.
 *
 * `silicon_tier` is `null` rather than zero when the model is not enrolled for
 * silicon at all, which is a different statement from being enrolled at the
 * lowest tier.
 */
export interface ModelReadiness {
  science_tier: number;
  science_label: string;
  silicon_tier: number | null;
  silicon_label: string;
  is_perfect?: boolean;
  validation?: Record<string, unknown>;
  silicon?: Record<string, unknown>;
  terminal_silicon_tier?: string;
  terminal_reason?: string;
}

/**
 * What the compiler will accept for this model.
 *
 * `cosim_integrators` is a subset of `integrators`: an integrator can be
 * compilable without a co-simulation reference to check it against.
 */
export interface ModelCompileConfiguration {
  schema_name: string;
  default_integrator: string;
  integrators: string[];
  cosim_integrators: string[];
  default_q_format: string;
  q_formats: string[];
}

/**
 * Everything needed to configure and run one model.
 *
 * `reproducibility` is the claim that matters: a reference configuration and
 * the digest of the trace it must produce. `golden_trace_sha256_variants`
 * exists because some models legitimately produce one of several traces
 * depending on the platform's transcendentals, and collapsing that to one
 * digest would report a false failure.
 */
export interface ModelDetail extends ModelSummary {
  docstring: string; display_name: string;
  state_vars: ModelStateVariable[];
  params: ModelParameter[];
  dynamics: Record<string, string>;
  backends: ModelBackendSupport[];
  readiness?: ModelReadiness;
  reproducibility: {
    reference_config: string;
    golden_trace_sha256: string;
    golden_trace_sha256_variants?: string[];
    reproducible: boolean;
  };
  documentation_slug: string;
  compile_configuration?: ModelCompileConfiguration | null;
}

/**
 * The facets the catalogue can be filtered by, counted by the server.
 *
 * Counted server-side so a facet the catalogue has stopped carrying disappears
 * from the filters with it, rather than staying as an empty option.
 */
export interface ModelFacets {
  /** Every registered identity, so this does not move when a descriptor breaks. */
  total: number;
  /**
   * Digest over the identities and their metadata states. Two clients holding
   * the same revision hold the same corpus in the same health.
   */
  corpus_revision: string;
  /** How many entries are in each metadata state. */
  metadata_states: Record<ModelMetadataState, number>;
  /** The entries whose metadata could not be read, by name. */
  invalid_models: string[];
  families: { family: string; category_slug: string; count: number }[];
  maturities: Record<string, number>;
  behaviors: { tag: string; count: number }[];
  /** Counts by science tier label (S0–S5). */
  science_tiers?: Record<string, number>;
  /** Counts by silicon label (none, H0–H5). */
  silicon_tiers?: Record<string, number>;
}

/** A model's prose documentation, as Markdown, with the slug it is filed under. */
export interface ModelDoc {
  name: string;
  slug: string;
  markdown: string;
}

/** A saved experiment, with the view it is best read in. */
export interface PresetSummary {
  id: string; title: string; description: string; suggested_view: string;
}

/**
 * What a sweep point's attractor set actually is.
 *
 * `insufficient_samples` is a verdict, not an absence: it says the point was
 * evaluated and could not be classified, which is different from not having
 * been evaluated.
 */
export type AttractorKind = "extrema" | "fixed_point" | "insufficient_samples";

/** A numerical extrema sweep under the configured drive; not a bifurcation continuation. */
export interface BifurcationResponse {
  param_name: string; param_values: number[]; attractors: number[][];
  attractor_kinds?: AttractorKind[];
  variable?: string | null;
  protocol?: string;
  contract?: MetricContract;
  analysis_metadata: AnalysisResultMetadata;
}

/**
 * How much one parameter moves the firing rate, or why that is undefined.
 *
 * Elasticity is undefined at a zero base rate or a zero parameter; those
 * points carry `null` and a `reason` rather than a zero that would plot as an
 * insensitive parameter.
 */
export interface SensitivityRow {
  param: string;
  /** Rate elasticity, or null where it is undefined (see ``reason``). */
  sensitivity: number | null;
  base_rate?: number;
  reason?: string;
  rate_minus?: number;
  rate_plus?: number;
}

/** Every parameter's elasticity around one base rate. */
export interface SensitivityResponse {
  analysis_metadata: AnalysisResultMetadata;
  contract?: MetricContract;
  base_rate: number;
  sensitivities: SensitivityRow[];
}

/**
 * How far one variable drifted between two runs of the same experiment.
 *
 * `first_divergence_step` is `null` when the two never separated by more than
 * `divergence_tolerance`; the tolerance travels with the figure so a
 * divergence step read later is read against the threshold it was found under.
 * `trace` is per raw step and `display` is at the reference's display samples,
 * so a plot and a claim about the maximum error do not disagree.
 */
export interface PrecisionVariableComparison {
  max_abs_error: number;
  mean_abs_error: number;
  rms_error: number;
  final_abs_error: number;
  first_divergence_step: number | null;
  divergence_tolerance: number;
  /** Full-resolution absolute error per raw step. */
  trace: number[];
  /** Error at the float result's display sample indices. */
  display: number[];
}

/**
 * Whether two runs spiked the same way, and where they first stopped agreeing.
 *
 * Counts alone hide a shifted train, so pairing and the largest paired offset
 * are reported too: identical counts with a non-zero offset is a real
 * difference.
 */
export interface PrecisionEventComparison {
  identical: boolean;
  reference_count: number;
  candidate_count: number;
  paired: number;
  max_paired_step_offset: number;
  first_divergence: { index: number; reference_step: number | null; candidate_step: number | null } | null;
}

/**
 * One candidate implementation measured against the float64 reference.
 *
 * `saturation` is present for fixed-point candidates: steps spent pinned at a
 * word boundary explain an error that otherwise looks like drift.
 */
export interface PrecisionCandidateComparison {
  candidate: string;
  variables: Record<string, PrecisionVariableComparison>;
  events: PrecisionEventComparison;
  saturation?: {
    max_word: number;
    min_word: number;
    per_variable: Record<string, { steps_at_max: number; steps_at_min: number }>;
  };
}

/** One value as requested, as encoded, and the error between the two. */
export interface PrecisionEncodingRow {
  requested: number;
  word: number;
  quantised: number;
  abs_error: number;
}

/**
 * Float64 reference versus (a) the bit-true fixed-point kernel run and (b) a
 * float64 run with quantised parameters; ``error`` summarises (a) for the
 * first declared variable.
 */
export interface PrecisionResponse {
  analysis_metadata: AnalysisResultMetadata;
  schema_version?: "studio.precision-compare.v2";
  contract?: MetricContract;
  float_result: SimulateResponse;
  fixed_result: SimulateResponse;
  parameter_quantisation_result?: SimulateResponse;
  arithmetic?: Record<string, unknown> & { q_format: string; overflow: string; rounding: string; compiler?: string };
  encoding?: {
    q_format: string;
    data_width: number;
    fraction: number;
    resolution: number;
    params: Record<string, PrecisionEncodingRow>;
    init: Record<string, PrecisionEncodingRow>;
    dt: PrecisionEncodingRow;
    drive: { protocol: string; frequency_hz: number; n_steps: number; max_abs_error: number; min: number; max: number };
  };
  comparison?: {
    bit_true: PrecisionCandidateComparison;
    parameter_quantisation: PrecisionCandidateComparison;
  };
  error: {
    kind?: string;
    variable: string;
    max_error: number;
    mean_error: number;
    rms_error: number;
    first_divergence_step?: number | null;
    trace: number[];
    display?: number[];
  };
  quantized_params: Record<string, number>;
  quantized_init?: Record<string, number>;
}

/**
 * The nullclines of a two-variable system, with where they could be evaluated.
 *
 * `validity_0` and `validity_1` mark the cells the components could actually
 * be evaluated in; a curve drawn without them would run confidently through
 * regions where nothing was computed. `held` records what the other variables
 * were fixed at, without which the curves are not reproducible.
 */
export interface NullclineResponse {
  analysis_metadata: AnalysisResultMetadata;
  schema_version?: "studio.nullclines.v2";
  contract?: MetricContract;
  var_names: string[];
  nullcline_0: { variable: string; points: number[][]; cells?: number };
  nullcline_1: { variable: string; points: number[][]; cells?: number };
  grid?: { x: number[]; y: number[]; size: number };
  /** 1 where the component could be evaluated, 0 where it could not; rows follow y, columns x. */
  validity_0?: number[][];
  validity_1?: number[][];
  held?: Record<string, number>;
  current?: number;
  domain?: { status: MetricDomain; invalid_fraction: Record<string, number> };
}

/** Two runs of the same shape, for the side-by-side view. */
export interface CompareResponse {
  a: SimulateResponse;
  analysis_metadata: AnalysisResultMetadata;
  b: SimulateResponse;
}

/** Firing rate against input frequency, at one drive amplitude. */
export interface FreqResponse {
  analysis_metadata: AnalysisResultMetadata;
  frequencies_hz: number[];
  rates: number[];
  amplitude: number;
}

/**
 * One thing a capability needs, and whether this deployment has it.
 *
 * `detail` says what is missing rather than only that something is, so an
 * operator can act on it without reading the server's logs.
 */
export interface CapabilityRequirement {
  name: string;
  available: boolean;
  detail: string;
}

/**
 * What one Studio capability is and whether this deployment can offer it.
 *
 * `status` and `healthy` are not the same question: a capability can be
 * `stable` in the product and unhealthy in this deployment, and the Studio
 * needs both to decide between hiding a feature and showing it as broken.
 */
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

/** Every capability this deployment reports on, in one answer. */
export interface StudioCapabilitiesResponse {
  capabilities: StudioCapability[];
}

/**
 * Whether the audit log is configured, reachable, and currently working.
 *
 * `configured` and `healthy` are separate: an audit log that was never set up
 * is a deployment choice, and one that was set up and is failing is an
 * incident.
 */
export interface StudioAuditStatus {
  configured: boolean;
  healthy: boolean;
  last_error: string | null;
  path_configured: boolean;
  sink_type: string;
}

/**
 * One audit record, chained to the one before it.
 *
 * `previous_event_hash` and `event_hash` make the log tamper-evident: a
 * removed or edited record breaks the chain at the next one. Both are nullable
 * because a sink that does not hash still records events, and reporting the
 * absence is better than implying a chain that is not there.
 */
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

/**
 * A page of audit records, saying plainly when it is not all of them.
 *
 * `truncated` exists so an export is never read as a complete log by
 * accident.
 */
export interface StudioAuditExport {
  configured: boolean;
  event_count: number;
  events: StudioAuditEvent[];
  schema_version: string;
  sink_type: string;
  truncated: boolean;
}

/**
 * What one quarantine archive contains and why its records were quarantined.
 *
 * Quarantined records are ones the audit log could not accept as they stood;
 * `reason_counts` says what was wrong with them in aggregate, which is what an
 * operator needs before deciding to restore or purge.
 */
export interface StudioAuditQuarantineArchiveSummary {
  archive_artifact_count: number;
  event_count: number;
  quarantine_reason: string;
  reason_counts: Record<string, number>;
  retained_event_count: number;
  source_schema_version: string;
  truncated: boolean;
}

/** A created quarantine archive, its artefacts and the manifest that validates it. */
export interface StudioAuditQuarantineArchiveResult {
  archive_id: string;
  artifact_paths: string[];
  artifacts: StudioJobArtifact[];
  job_id: string;
  manifest: Record<string, unknown>;
  schema_version: string;
  summary: StudioAuditQuarantineArchiveSummary;
}

/**
 * What the server could and could not confirm about an archive.
 *
 * `errors` and `warnings` are kept apart because a missing manifest is not the
 * same as a manifest that disagrees with the archive, and only the second is a
 * reason to refuse it.
 */
export interface StudioAuditQuarantineArchiveValidation {
  archive_id: string | null;
  errors: string[];
  schema_version: string;
  summary: StudioAuditQuarantineArchiveSummary | null;
  valid: boolean;
  warnings: string[];
}

/** An archive's summary, plus what the restore itself did and when. */
export interface StudioAuditQuarantineArchiveRestoreSummary extends StudioAuditQuarantineArchiveSummary {
  restore_artifact_count: number;
  restored_at_utc: string;
}

/** A completed restore, with the artefacts it wrote back. */
export interface StudioAuditQuarantineArchiveRestoreResult {
  archive_id: string;
  artifact_paths: string[];
  artifacts: StudioJobArtifact[];
  job_id: string;
  manifest: Record<string, unknown>;
  schema_version: string;
  summary: StudioAuditQuarantineArchiveRestoreSummary;
}

/**
 * One archive in a retention plan, and what the plan would do with it.
 *
 * `disposition` is a proposal, not an action: the plan is read before the
 * purge that carries it out.
 */
export interface StudioAuditQuarantineArchiveRetentionEntry {
  archive_id: string;
  artifact_paths: string[];
  created_at_utc: string;
  disposition: "retain" | "prune_candidate";
  event_count: number;
  finished_at_utc: string | null;
  job_id: string;
  retained_event_count: number;
  summary: StudioAuditQuarantineArchiveSummary;
}

/**
 * What a purge would remove and keep, without removing anything.
 *
 * `skipped_record_count` names archives the plan could not classify, so a
 * purge is never read as covering more than it did.
 */
export interface StudioAuditQuarantineArchiveRetentionPlan {
  archive_count: number;
  entries: StudioAuditQuarantineArchiveRetentionEntry[];
  prune_candidate_count: number;
  retain_count: number;
  retain_latest: number;
  schema_version: string;
  skipped_record_count: number;
}

/** What a purge actually removed and kept, in the plan's own terms. */
export interface StudioAuditQuarantineArchivePurgeResult {
  purged_archive_count: number;
  purged_entries: StudioAuditQuarantineArchiveRetentionEntry[];
  retained_archive_count: number;
  retained_entries: StudioAuditQuarantineArchiveRetentionEntry[];
  retain_latest: number;
  schema_version: string;
  skipped_record_count: number;
}

/**
 * The job runner's health and its current load.
 *
 * `allowed_kinds` is part of the status rather than a separate lookup: a kind
 * the runner will not accept should not be offered.
 */
export interface StudioJobStatus {
  active_count: number;
  allowed_kinds: string[];
  completed_count: number;
  configured: boolean;
  failed_count: number;
  process_count: number;
  resource_profiles: StudioJobResourceProfile[];
  schema_version: string;
  thread_count: number;
  timed_out_count: number;
}

/** The limits one kind of job runs under, and how it may be executed. */
export interface StudioJobResourceProfile {
  default_timeout_seconds: number;
  execution_models: string[];
  kind: string;
  max_artifact_bytes: number;
}

/**
 * One file a job produced, with its digest.
 *
 * The digest travels with the path so an artefact downloaded later can be
 * shown to be the one the job wrote.
 */
export interface StudioJobArtifact {
  relative_path: string;
  sha256: string;
  size_bytes: number;
}

/**
 * One job: what it is, who owns it, where it got to, and what it left behind.
 *
 * `cancelling` is distinct from `cancelled` because a job that has been asked
 * to stop has not necessarily stopped, and treating the two as one loses the
 * only window in which that matters.
 */
export interface StudioJobRecord {
  artifacts: StudioJobArtifact[];
  created_at_utc: string;
  error: string | null;
  execution_model: "thread" | "process";
  finished_at_utc: string | null;
  job_id: string;
  kind: string;
  owner: string;
  request_id: string | null;
  result: Record<string, unknown> | null;
  started_at_utc: string | null;
  status: "pending" | "running" | "completed" | "failed" | "cancelling" | "cancelled" | "timed_out";
}

/** Every job the runner is holding. */
export interface StudioJobListResponse {
  jobs: StudioJobRecord[];
  schema_version: string;
}

/**
 * What to gather into an evidence bundle.
 *
 * Every category is passed explicitly rather than inferred from a project,
 * because a bundle is a claim about what was included, and a bundle that
 * quietly gathered more or less than was asked for would not be one.
 */
export interface StudioEvidenceBundleRequest {
  audit_limit: number;
  analysis_results: Record<string, unknown>[];
  command_replay: Record<string, unknown> | null;
  default_flow_attestations: Record<string, unknown>[];
  default_flow_runs: Record<string, unknown>[];
  include_audit: boolean;
  job_ids: string[];
  model_scan_results: Record<string, unknown>[];
  project_name: string | null;
  simulation_results: Record<string, unknown>[];
  weight_restore_results: Record<string, unknown>[];
  weight_restore_attach_results: Record<string, unknown>[];
}

/** What a bundle turned out to contain, counted by kind and by classification. */
export interface StudioEvidenceBundleSummary {
  artifact_path_count: number;
  entry_count: number;
  entry_type_counts: Record<string, number>;
  evidence_classification_counts: Record<string, number>;
  source_job_count: number;
  source_job_kind_counts: Record<string, number>;
  source_job_owner_counts: Record<string, number>;
}

/** A created evidence bundle, its artefacts and the manifest that identifies it. */
export interface StudioEvidenceBundleResponse {
  artifact_paths: string[];
  artifacts: StudioJobArtifact[];
  bundle_id: string;
  job_id: string;
  manifest: Record<string, unknown>;
  schema_version: string;
  summary: StudioEvidenceBundleSummary;
}

/**
 * The capability picture in one line of counts, for the operator view.
 *
 * `healthy_count` is counted separately from the status counts because a
 * capability's declared status and its health in this deployment are different
 * questions.
 */
export interface StudioOperatorCapabilityStatus {
  degraded_count: number;
  experimental_count: number;
  healthy_count: number;
  stable_count: number;
  total_count: number;
  unavailable_count: number;
}

/**
 * How this deployment authenticates, and whether a weaker mode is permitted.
 *
 * `header_principal_allowed` is surfaced on its own because trusting a header
 * is only safe behind a proxy that sets it, and an operator needs to see that
 * it is on without inferring it from the mode.
 */
export interface StudioOperatorIdentityStatus {
  configured: boolean;
  header_principal_allowed: boolean;
  mode: "service_account" | "header_principal" | "disabled";
}

/**
 * How much of the route surface is actually protected, and whether it is enforced.
 *
 * `enforced` and the counts are both needed: policies that exist but are not
 * enforced are the failure this reports, and counts alone would look healthy.
 */
export interface StudioOperatorRoutePolicyStatus {
  admin_count: number;
  authenticated_count: number;
  enforced: boolean;
  protected_audit_action_count: number;
  protected_count: number;
  protected_routes_audited: boolean;
  public_count: number;
  total_count: number;
}

/**
 * The limits jobs and external tools run under.
 *
 * `eda_process_limits_supported` is reported because the CPU and memory limits
 * are only meaningful on hosts that expose them; elsewhere they are `null` and
 * saying so is better than showing an unenforced number.
 */
export interface StudioOperatorResourceLimitStatus {
  eda_process_cpu_seconds: number | null;
  eda_process_memory_bytes: number | null;
  eda_process_limits_supported: boolean;
  job_default_timeout_seconds: number;
  job_max_artifact_bytes: number;
}

/** The sign-in throttle: its thresholds, and how many buckets are active or locked. */
export interface StudioOperatorBrowserLoginStatus {
  active_bucket_count: number;
  cooldown_seconds: number;
  failure_window_seconds: number;
  locked_bucket_count: number;
  max_retry_after_seconds: number;
  max_failures: number;
}

/**
 * Everything an operator needs to judge a deployment, in one answer.
 *
 * `deployment_profile` is here so the rest is read in context: a development
 * profile is expected to be permissive, and the same figures under a
 * production profile are findings.
 */
export interface StudioOperatorStatus {
  audit: StudioAuditStatus;
  browser_login: StudioOperatorBrowserLoginStatus;
  capabilities: StudioOperatorCapabilityStatus;
  deployment_profile: "development" | "production";
  identity: StudioOperatorIdentityStatus;
  jobs: StudioJobStatus;
  resource_limits: StudioOperatorResourceLimitStatus;
  route_policies: StudioOperatorRoutePolicyStatus;
  schema_version: string;
}

/** A machine principal: its roles, whether it is active, and when it lapses. */
export interface StudioIdentityServiceAccount {
  active: boolean;
  expires_at_utc: string | null;
  principal_id: string;
  roles: string[];
}

/** Every service account this deployment holds. */
export interface StudioIdentityServiceAccountsResponse {
  schema_version: string;
  service_accounts: StudioIdentityServiceAccount[];
}

/**
 * A person who signs in, as the server describes them.
 *
 * No credential material appears here, and none is ever returned by the
 * identity routes.
 */
export interface StudioIdentityBrowserUser {
  active: boolean;
  expires_at_utc: string | null;
  principal_id: string;
  roles: string[];
  username: string;
}

/** Every browser user this deployment holds. */
export interface StudioIdentityBrowserUsersResponse {
  browser_users: StudioIdentityBrowserUser[];
  schema_version: string;
}

/** The fields a service account may be changed through. */
export interface StudioIdentityServiceAccountUpdate {
  active: boolean;
  expires_at_utc: string | null;
  roles: string[];
}

/**
 * The fields a browser user may be changed through.
 *
 * The password is deliberately absent: rotating a credential has its own route
 * so it is never a side effect of editing roles or expiry.
 */
export interface StudioIdentityBrowserUserUpdate {
  active: boolean;
  expires_at_utc: string | null;
  roles: string[];
}

/** A new browser user, with the one credential that is sent at creation. */
export interface StudioIdentityBrowserUserCreate {
  active: boolean;
  expires_at_utc: string | null;
  password: string;
  principal_id: string;
  roles: string[];
  username: string;
}

/** A password rotation: the new credential and nothing else. */
export interface StudioIdentityBrowserUserPasswordRotate {
  password: string;
}

/** Who the current token signs in as, and what it may do. */
export interface StudioAuthSession {
  authenticated: boolean;
  principal_id: string | null;
  roles: string[];
}

/**
 * A successful sign-in: the token, when it lapses, and what it grants.
 *
 * The expiry is returned so a client can act before a request fails rather
 * than discovering the lapse as a rejection.
 */
export interface StudioLoginResponse {
  access_token: string;
  expires_at_utc: string;
  principal_id: string;
  roles: string[];
  token_type: "bearer";
}

/** Whether the server actually revoked the session it was asked to end. */
export interface StudioLogoutResponse {
  revoked: boolean;
}

/**
 * One DCLS backend's availability and, when it ran, what it produced.
 *
 * `available` and `live` are separate: a backend can be installed and not
 * reachable. `bit_exact` and `parity` describe agreement with the reference,
 * which is what makes a backend usable as evidence rather than merely present.
 */
export interface DclsBackendStatus {
  backend: string;
  available: boolean;
  live: boolean;
  output_q88?: number;
  output?: number;
  bit_exact?: boolean;
  parity?: string;
}

/**
 * What the DCLS implementation is: its provenance, its arithmetic, its backends.
 *
 * The fixed-point block is stated rather than assumed because a weight format
 * read differently at either end is a silent numerical error, not a failure.
 */
export interface DclsInfo {
  name: string;
  provenance: { authors: string[]; year: number; venue: string; title: string; doi: string };
  fixed_point: {
    weight_format: string; accumulator_format: string;
    one: number; fraction_bits: number; parity: string;
  };
  backends: { backend: string; available: boolean; live: boolean }[];
  backend_order: string[];
  rtl_modules: string[];
  synthesis_target: string;
}

/**
 * One DCLS evaluation: the kernel it built and what each backend made of it.
 *
 * Every quantity appears twice, once in Q8.8 and once as a real number, so a
 * disagreement between backends can be read in the arithmetic it happened in
 * rather than after conversion.
 */
export interface DclsEvaluation {
  profile: {
    centre_q88: number; sigma_q88: number; centre: number; sigma: number;
    n_taps: number; gates_q88: number[]; gates: number[];
  };
  forward: {
    reference_output_q88: number; reference_output: number;
    active_tap_count: number; max_gate_q88: number; overflow: boolean;
    backends: DclsBackendStatus[]; bit_exact: boolean;
  };
}

/** What to evaluate: the kernel's shape, and optionally the input to drive it with. */
export interface DclsEvaluateBody {
  centre_q88: number;
  sigma_q88: number;
  n_taps: number;
  spikes?: number[];
  weights_q88?: number[];
}

/**
 * A recorded DCLS benchmark, with what it was measured on.
 *
 * `hardware_measurement_claimed` is stated rather than implied: a figure taken
 * on a simulated backend is not a hardware measurement, and reporting it as
 * one would be the difference between a benchmark and a claim.
 */
export interface DclsBenchmark {
  date_utc: string;
  cpu: string;
  workload: { n_channels: number; n_taps: number; elements: number; spike_density: number };
  isolation_mode: string;
  hardware_measurement_claimed: boolean;
  backends: {
    backend: string; median_call_ms: number;
    channels_per_s: number; speedup_over_python: number;
  }[];
}

/**
 * A measurement shaped for the shared databank.
 *
 * It carries its own environment and its parity against the reference, because
 * a speed-up from a backend that is not bit-exact is not comparable with one
 * from a backend that is, and a leaderboard without that distinction ranks the
 * wrong thing.
 */
export interface BenchmarkSubmission {
  schema_version: string;
  kernel: string;
  workload: { n_channels: number; n_taps: number; elements: number; spike_density: number };
  backends: {
    backend: string; median_call_ms: number; channels_per_s: number;
    speedup_over_python: number; repeats: number; bit_exact: boolean;
  }[];
  parity: { reference: string; tolerance: number; bit_exact_all: boolean };
  environment: { cpu: string; os: string; python: string; numpy: string; toolchains: Record<string, string> };
  hardware_measurement_claimed: boolean;
  contributor: { handle: string };
}

/** The contributed benchmarks, reduced to what the leaderboard shows. */
export interface DatabankLeaderboard {
  count: number;
  entries: {
    cpu: string; handle: string; fastest_backend: string;
    speedup: number; workload: { n_channels: number; n_taps: number };
  }[];
}

/** The analyses that run as jobs rather than inside a request. */
export type AnalysisJobKind = "fi_curve" | "bifurcation" | "heatmap" | "sensitivity";

/**
 * An analysis to queue, and the payload it is queued with.
 *
 * The payload is opaque here: each analysis has its own request shape, and
 * flattening them into one would let a field meant for one analysis be sent
 * with another.
 */
export interface AnalysisJobRequestBody {
  analysis: AnalysisJobKind;
  payload: Record<string, unknown>;
}

/**
 * What came back from queueing an analysis.
 *
 * `status_route` is where to look rather than a path to reassemble, and
 * `projected_simulations` says how much work was queued, so a caller can tell
 * a slow job from a stuck one.
 */
export interface AnalysisJobReceipt {
  analysis: AnalysisJobKind;
  dt_ms?: number;
  duration_ms?: number;
  execution_mode: "async_job";
  job: StudioJobRecord;
  job_id: string;
  projected_simulations?: number;
  schema_version: "studio.analysis.job.v1";
  status_route: string;
}

/** Whichever analysis result a completed job carries. */
export type AnalysisJobResult =
  | FICurveResponse
  | BifurcationResponse
  | HeatmapResponse
  | SensitivityResponse;

/** What came back from starting a catalogue scan, and where to follow it. */
export interface ModelScanJobReceipt {
  execution_mode: "async_job";
  job: StudioJobRecord;
  job_id: string;
  schema_version: "studio.model-scan.job.v1";
  status_route: string;
}

/**
 * A balanced network run: its raster and its population rates.
 *
 * The raster is two parallel arrays rather than pairs because it is drawn a
 * hundred thousand points at a time, and the rates are on their own coarser
 * clock, which is why `rate_time` is separate from the spike times.
 */
export interface NetworkResult {
  spike_times: number[];
  spike_neurons: number[];
  n_exc: number; n_inh: number; n_total: number; n_spikes: number;
  rate_time: number[]; exc_rates: number[]; inh_rates: number[];
  duration: number; dt: number;
  mean_exc_rate: number; mean_inh_rate: number;
}

/**
 * What one model did during a catalogue scan.
 *
 * `error_type` is present when the model was reached but misbehaved, which is
 * a different outcome from the model failing to run at all.
 */
export interface ModelBehavior {
  name: string; category: string; pattern: string;
  description: string; rate_hz: number; spike_count: number;
  error_type?: string;
}

/** A model the scan could not run, and what stopped it. */
export interface ModelScanFailure {
  name: string; category: string; error_type: string; error_message: string;
}

/**
 * What a catalogue scan was, and what it could not cover.
 *
 * `failed_models` is carried in full rather than as a count, so a scan is
 * never read as covering the catalogue when part of it did not run.
 */
export interface ModelScanMetadata {
  current: number;
  duration: number;
  error_count: number;
  evidence_classification: "analysis";
  failed_models: ModelScanFailure[];
  input_sha256: string;
  model_count: number;
  pattern_counts: Record<string, number>;
  result_sha256: string;
  schema_version: "studio.model-scan.v1";
  status: "completed";
}

/** A catalogue scan's findings, with the metadata that identifies it. */
export interface ModelScanResponse {
  models: ModelBehavior[];
  scan_metadata: ModelScanMetadata;
  schema_version: "studio.model-scan.v1";
}

/**
 * A model's character: how it fires, what drives it, and what moves it.
 *
 * `threshold_current` is `null` when the sweep never elicited a spike, which
 * is a finding about the model rather than a missing value.
 */
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

/**
 * A recorded trace the server has taken in, with what it derived from it.
 *
 * `threshold_estimate` is the server's own guess at where the spikes were
 * detected, so an overlay can be read against the same threshold the spike
 * count came from.
 */
export interface ImportedTrace {
  time: number[];
  voltage: number[];
  spikes: number[];
  spike_count: number;
  dt: number;
  stats: { mean: number; std: number; min: number; max: number; threshold_estimate: number };
}

/** The custom system a compilation was asked to lower. */
export interface CompileSourcePayload {
  equations: string[];
  init: Record<string, number>;
  params: Record<string, number>;
  reset: string | null;
  threshold: string | null;
}

/**
 * The catalogue model a compilation was asked to lower, pinned exactly.
 *
 * `schema_sha256` pins the descriptor the model was read from, so RTL can be
 * traced to a version of the model and not merely to its name.
 */
export interface ModelCompileSourcePayload {
  dt: number;
  integrator: string;
  model_name: string;
  params: Record<string, number>;
  q_format: string;
  schema_name: string;
  schema_sha256: string;
}

/** What to compile: the model, its parameters, and the arithmetic to compile it in. */
export interface ModelCompileRequest {
  model_name: string;
  params: Record<string, number>;
  dt: number;
  integrator: string;
  q_format: string;
  module_name?: string;
}

/** A compilation plus the stimulus to co-simulate the result against. */
export interface ModelCosimRequest extends ModelCompileRequest {
  current: number;
  n_steps: number;
}

/** The configuration a compilation actually ran under, as against what was asked. */
export interface ModelCompileResultConfiguration {
  dt: number;
  integrator: string;
  model_name: string;
  q_format: string;
  schema_name: string;
  schema_sha256: string;
}

/** The RTL a compilation produced, identified by digest rather than by name. */
export interface CompileTraceabilityOutput {
  language: "verilog" | "systemverilog";
  module_name: string;
  rtl_chars: number;
  rtl_sha256: string;
}

/**
 * The chain from a source system to a specific piece of RTL.
 *
 * `input_sha256`, the source payload and `output.rtl_sha256` together are what
 * let a later step say that this RTL came from that model under those
 * settings. A synthesis result without it says what was built but not what it
 * was built from.
 */
export interface CompileTraceability {
  evidence_classification: "compile";
  input_sha256: string;
  output: CompileTraceabilityOutput;
  schema_version: "studio.compile-traceability.v1";
  source: "ode" | "model";
  source_payload: CompileSourcePayload | ModelCompileSourcePayload;
  status: "completed";
  traceability_sha256: string;
}

/** Generated RTL, with the traceability that says where it came from. */
export interface CompileResponse {
  chars: number;
  compile_configuration?: ModelCompileResultConfiguration;
  compile_traceability: CompileTraceability;
  module_name: string;
  verilog: string;
}

/**
 * The first cycle at which RTL and reference disagreed, and on what.
 *
 * Both sides' values are carried, not just the difference: which one is wrong
 * is not decidable from a delta.
 */
export interface ModelCosimMismatch {
  cycle: number;
  reference: Record<string, number>;
  rtl: Record<string, number>;
  signals: string[];
}

/**
 * Whether a model's RTL reproduces its reference implementation exactly.
 *
 * `bit_exact` is the claim; everything else is what makes it checkable — both
 * sides identified by digest, the stimulus they were driven with, and the
 * exact tool versions that ran them. `first_mismatch` is `null` only when
 * there was none.
 */
export interface ModelCosimReport {
  bit_exact: boolean;
  configuration: ModelCompileResultConfiguration;
  first_mismatch: ModelCosimMismatch | null;
  module_name: string;
  reference: { kind: "generated_bit_true_c"; source_sha256: string; trace_sha256: string };
  rtl: { kind: "iverilog_vvp"; source_sha256: string; trace_sha256: string };
  sample_count: number;
  schema_version: "studio.cosim-parity.v1";
  signals: string[];
  status: "completed";
  stimulus: { current: number; current_q: number; n_steps: number };
  tools: Record<"gcc" | "iverilog" | "vvp", string>;
}

/**
 * A built IR document, its shape, and whatever refused to build.
 *
 * `errors` is non-empty alongside an IR when the build partially succeeded, so
 * a caller reads the errors rather than assuming the text is complete.
 */
export interface IRBuildResponse {
  ir_text: string;
  errors: string[];
  n_ops: number;
  n_inputs: number;
  n_outputs: number;
  graph_name: string;
  params_q88: Record<string, number>;
}

/** Whether an IR document holds the invariants the emitters rely on. */
export interface IRVerifyResponse {
  valid: boolean;
  errors: string[];
  n_ops: number;
  graph_name: string;
}

/** SystemVerilog emitted from an IR document. */
export interface SVEmitResponse {
  systemverilog: string;
  graph_name: string;
  chars: number;
}

/**
 * Verilog emitted straight from a system, with the IR it went through.
 *
 * The IR is returned even though it was not asked for, so a direct emission is
 * as inspectable as the two-step route it replaces.
 */
export interface SVDirectResponse {
  verilog: string;
  ir_repr: string;
  chars: number;
  module_name: string;
  compile_traceability: CompileTraceability;
}

/**
 * Whether one synthesis tool is present, and which version.
 *
 * The version is `null` when the tool is absent, rather than an empty string,
 * so "not installed" and "version could not be read" stay distinguishable.
 */
export interface SynthToolInfo {
  available: boolean;
  version: string | null;
}

/** What a design consumed, in the units the toolchain reports. */
export interface SynthResources {
  luts: number;
  ffs: number;
  brams: number;
  dsps: number;
  cells: number;
  wires: number;
}

/** What the target device holds, so consumption can be read as a fraction. */
export interface SynthCapacity {
  luts: number;
  ffs: number;
  brams: number;
  dsps: number;
}

/** One tool in a target's toolchain: what it is, what it does, and whether it is there. */
export interface SynthesisToolProvenance {
  available: boolean;
  executable: string;
  key: string;
  role: "synthesis" | "place_and_route";
  version: string | null;
}

/**
 * What a target's figures rest on.
 *
 * `provenance_grade` is the load-bearing field: `tool_backed` means the
 * capacity and readiness came from the toolchain itself, `unverified` means
 * they came from a table. A utilisation percentage means something quite
 * different under each, and collapsing them would let a table entry read as a
 * measurement.
 */
export interface SynthesisTargetProvenance {
  capacity: Partial<SynthCapacity>;
  device: string | null;
  evidence_classification: "synthesis";
  pnr_ready: boolean;
  pnr_tool: string | null;
  provenance_grade: "tool_backed" | "unverified";
  schema_version: "studio.synthesis-target-provenance.v1";
  status: "completed";
  synthesis_command: string;
  synthesis_ready: boolean;
  target: string;
  tools: SynthesisToolProvenance[];
}

/**
 * Every target's provenance in one document, identified by digest.
 *
 * The matrix carries a single grade as well: a comparison across targets is
 * only as good as its weakest provenance, and saying so once prevents a
 * tool-backed target lending credibility to a tabled one beside it.
 */
export interface SynthesisTargetProvenanceMatrix {
  evidence_classification: "synthesis";
  matrix_sha256: string;
  provenance_grade: "tool_backed" | "unverified";
  schema_version: "studio.synthesis-target-provenance-matrix.v1";
  status: "completed";
  targets: Record<string, SynthesisTargetProvenance>;
}

/**
 * One synthesis run: what it used, what the device holds, and what backs it.
 *
 * `utilisation` is derived from `resources` and `capacity` but is carried
 * explicitly, so the figure shown is the one the server computed rather than
 * one recomputed in the browser against a capacity it may have misread.
 */
export interface SynthResult {
  success: boolean;
  error?: string;
  target: string;
  resources: SynthResources;
  capacity: SynthCapacity;
  utilisation: Record<string, number>;
  log_excerpt: string;
  target_provenance: SynthesisTargetProvenance;
  silicon_terminal?: SiliconTerminalResult;
}

/**
 * A resource estimate made without synthesising.
 *
 * `estimated` is always true here and is carried anyway, so a result that
 * travels alongside a real synthesis cannot be mistaken for one.
 */
export interface SynthEstimate {
  target: string;
  estimated: boolean;
  resources: { luts: number; ffs: number; brams: number; dsps: number };
  capacity: SynthCapacity;
  utilisation: Record<string, number>;
}

/** The same design synthesised for every supported target, with their provenance. */
export interface MultiTargetResult {
  targets: Record<string, SynthResult>;
  supported: string[];
  target_provenance_matrix: SynthesisTargetProvenanceMatrix;
}

/**
 * A place-and-route outcome.
 *
 * The frequency and critical path are nullable rather than absent on failure,
 * so a failed route reports the fields it could not fill instead of a shape
 * the caller has to guess at.
 */
export interface PnRResult {
  success: boolean;
  error?: string;
  max_freq_mhz?: number | null;
  critical_path?: string | null;
  log_excerpt?: string;
}

/**
 * Every digest linking a piece of silicon back to the model it came from.
 *
 * This is the chain a terminal result asserts: the compile input, its
 * traceability, both co-simulation traces, and the RTL. Any one of them
 * missing would leave the claim unfalsifiable.
 */
export interface SiliconTerminalSourceChain {
  compile_input_sha256: string;
  compile_traceability_sha256: string;
  cosim_reference_trace_sha256: string;
  cosim_rtl_trace_sha256: string;
  model_name: string;
  module_name: string;
  rtl_sha256: string;
}

/**
 * The end of the pipeline: a routed design and the chain that produced it.
 *
 * `status` and `success` are both present because a run can complete and fail,
 * and a caller reporting on evidence needs to distinguish "did not finish"
 * from "finished and did not work".
 */
export interface SiliconTerminalResult {
  artifacts: {
    netlist_sha256: string | null;
    routed_design_sha256: string | null;
  };
  evidence_classification: "synthesis";
  place_and_route: PnRResult | null;
  schema_version: "studio.silicon-terminal.v1";
  source_chain: SiliconTerminalSourceChain;
  status: "completed" | "failed";
  success: boolean;
  synthesis: SynthResult;
  target: string;
  target_provenance: SynthesisTargetProvenance;
}

/** One surrogate gradient, and whether this deployment has it. */
export interface SurrogateInfo {
  name: string;
  available: boolean;
}

/** One trainable cell type, and whether this deployment has it. */
export interface CellTypeInfo {
  name: string;
  available: boolean;
}

/**
 * Everything a training run is defined by.
 *
 * `learn_beta` and `learn_threshold` decide whether the neuron's own constants
 * are trained alongside the weights, which changes what a checkpoint contains
 * and therefore what it can be restored into.
 */
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

/** One epoch's losses and accuracies, on both the training and validation sets. */
export interface TrainingEpochMetrics {
  epoch: number;
  train_loss: number;
  train_accuracy: number;
  val_loss: number;
  val_accuracy: number;
  layer_spike_rates: Record<string, number>;
  param_snapshot: Record<string, number>;
}

/** Where a training run got to, and its final numbers if it finished. */
export interface TrainingJobStatus {
  job_id: string;
  status: string;
  error: string | null;
  final_metrics: Record<string, number> | null;
}

/**
 * A training checkpoint: what was trained, under which config, and its digest.
 *
 * `config_sha256` is what makes a restore safe — weights only fit a network
 * built the same way, and the digest is how that is checked rather than
 * assumed.
 */
export interface TrainingCheckpointPayload {
  checkpoint_sha256: string;
  config: Partial<TrainingConfig>;
  config_sha256: string;
  evidence_summary: Record<string, unknown> | null;
  final_metrics: Record<string, number> | null;
  generated_at_utc: string;
  job_id: string;
  schema_version: "studio.training.checkpoint.v1";
  status: string;
  weight_checkpoint?: TrainingWeightCheckpoint | null;
}

/** One stored weight file, with the digest that identifies it. */
export interface TrainingWeightArtifact {
  relative_path: string;
  sha256: string;
  size_bytes: number;
}

/**
 * Where a run's weights live and what shape they are in.
 *
 * Most fields are optional because a checkpoint written by an older run
 * carries fewer of them; the weights artefact is not, because a checkpoint
 * without it is not one.
 */
export interface TrainingWeightCheckpoint {
  architecture?: string;
  config_sha256?: string;
  final_metrics?: Record<string, unknown> | null;
  format?: string;
  framework?: string;
  metadata_artifact?: TrainingWeightArtifact;
  parameter_count?: number;
  schema_version: "studio.training.weight-checkpoint.v1";
  weights_artifact: TrainingWeightArtifact;
}

/**
 * How to fetch and verify a run's weights before restoring them.
 *
 * The route template and the loader policy are stated rather than assumed by
 * the client: the weights are fetched through the authenticated artefact route
 * and checked against their digest, and writing that down here is what stops a
 * caller inventing a shortcut around it.
 */
export interface TrainingWeightRestorePlan {
  architecture: string;
  artifact_route_template: "/api/studio/jobs/{job_id}/artifacts/{artifact_path}";
  config_sha256: string;
  format: string;
  framework: string;
  loader_policy: "download_from_authenticated_artifact_route_and_verify_sha256";
  metadata_artifact: TrainingWeightArtifact;
  parameter_count: number;
  restore_ready: boolean;
  schema_version: "studio.training.weight-restore-plan.v1";
  source_job_id: string;
  source_status: string;
  weights_artifact: TrainingWeightArtifact;
}

/**
 * What a restore actually loaded, as against what it planned to.
 *
 * `loaded_key_count` beside `parameter_count` is the check that matters: a
 * restore that loaded fewer keys than the checkpoint holds has silently left
 * part of the network at its initial values.
 */
export interface TrainingWeightMaterialization {
  architecture: string;
  config_sha256: string;
  format: string;
  framework: string;
  loaded_key_count: number;
  metadata_sha256: string;
  parameter_count: number;
  schema_version: "studio.training.weight-materialization.v1";
  source_job_id: string;
  weights_sha256: string;
}

/** A completed weight restore, with the artefacts it wrote and what it loaded. */
export interface TrainingWeightRestoreResult {
  artifacts: StudioJobArtifact[];
  evidence_classification: "training";
  job_id: string;
  materialization: TrainingWeightMaterialization;
  schema_version: "studio.training.weight-restore.v1";
  source_job_id: string;
  source_status: string;
  status: "completed";
}

/**
 * A new run started with another run's weights attached.
 *
 * `architecture_fingerprint` records what the weights were attached to, so a
 * later mismatch can be attributed rather than guessed at.
 */
export interface TrainingWeightAttachResult {
  architecture_fingerprint: string;
  job_id: string;
  source_job_id: string;
  status: string;
}

/** Weights attached to a run that was already going. */
export interface TrainingWeightLiveAttachResult {
  architecture_fingerprint: string;
  source_job_id: string;
  status: string;
  target_job_id: string;
}

/**
 * What the server made of an imported checkpoint.
 *
 * The weight checkpoint and the restore plan are both nullable: a checkpoint
 * can be readable as a configuration while carrying no usable weights, and
 * saying so is more useful than refusing the import outright.
 */
export interface TrainingCheckpointImportResponse {
  config: Partial<TrainingConfig>;
  config_sha256: string;
  imported_schema_version: "studio.training.checkpoint.v1";
  source_job_id: string;
  source_status: string;
  source_weight_checkpoint: TrainingWeightCheckpoint | null;
  weight_restore_plan: TrainingWeightRestorePlan | null;
}

/** One training run in a list: where it got to and what it is running. */
export interface TrainingJobSummary {
  job_id: string;
  status: string;
  config: TrainingConfig;
}

/** Which sign a population's outgoing weights take. */
export type StudioNeuronType = "excitatory" | "inhibitory";

/**
 * The external input a population receives, if any.
 *
 * A discriminated union rather than one shape with optional fields, so a
 * Poisson rate cannot be set on a constant drive and quietly ignored.
 */
export type PopulationDrive =
  | { kind: "none" }
  | { kind: "constant"; current: number }
  | { kind: "poisson"; rate_hz: number; weight: number; seed?: number };

/**
 * One population in the network graph.
 *
 * The identity is the server's; `position` is the canvas's and means nothing
 * to a simulation, which is why it travels with the node rather than being
 * inferred when the graph is drawn.
 */
export interface PopulationNode {
  id: string;
  type: "population";
  label: string;
  model: string;
  count: number;
  neuron_type: StudioNeuronType;
  position: { x: number; y: number };
  params: Record<string, number>;
  /** External input; a node saved before drives existed resolves to no input. */
  drive?: PopulationDrive;
}

/** How a projection connects its two populations. */
export type ProjectionRule = "random" | "all_to_all";

/**
 * One projection between two populations.
 *
 * `seed` and `autapses` are optional because the create route does not carry
 * them; an edge that needs either is created first and then given them, which
 * is why they can be absent on an edge the server has just returned.
 */
export interface ProjectionEdge {
  id: string;
  source: string;
  target: string;
  /** Signed synaptic weight: negative for an inhibitory source population. */
  weight: number;
  /** Delay in milliseconds; must be a whole number of graph timesteps. */
  delay: number;
  /** Connection rule; an edge saved before rules existed resolves to random. */
  rule?: ProjectionRule;
  /** Connection probability of the random rule; absent for all_to_all. */
  probability?: number;
  seed?: number;
  autapses?: boolean;
}

/**
 * The body `POST /api/graph/population` accepts.
 *
 * Not a `Partial<PopulationNode>`: the route takes a position as `x` and `y`
 * and has no use for `id`, `type` or `position`, so describing it as a partial
 * node both offered fields the route ignores and hid the two it needs.
 */
export interface PopulationCreateRequest {
  count: number;
  drive?: PopulationDrive;
  label: string;
  model: string;
  neuron_type: StudioNeuronType;
  params?: Record<string, number>;
  x: number;
  y: number;
}

/** One constructor field a population of this model may override. */
export interface PopulationModelParameter {
  default: number | null;
  kind: "float" | "int";
  name: string;
}

/** One field that is not an input, and the reason it is not. */
export interface PopulationModelUnsupported {
  name: string;
  reason: string;
}

/** What a population of one model may be given, as the server decides it. */
export interface PopulationModelContract {
  drive: { kind: "float" | "int"; parameter: string; positional_only: boolean };
  model: string;
  parameters: PopulationModelParameter[];
  schema_version: "studio.population-model-contract.v1";
  unsupported: PopulationModelUnsupported[];
}

/** One graph validation failure with the request field it came from. */
export interface GraphValidationIssue {
  field: string;
  message: string;
}

/** The answer of `POST /api/graph/validate`: every failure, each located. */
export interface GraphValidation {
  errors: string[];
  issues: GraphValidationIssue[];
  valid: boolean;
}

/**
 * A whole network: its populations, its projections, and how to run them.
 *
 * `duration`, `dt` and `seed` are optional because the same document is used
 * for validation, where none of them apply.
 */
export interface NetworkGraph {
  populations: PopulationNode[];
  projections: ProjectionEdge[];
  duration?: number;
  dt?: number;
  seed?: number;
}

/**
 * What one population did during a graph run.
 *
 * `offset` is where this population's neurons start in the global numbering,
 * which is what makes the raster's neuron indices interpretable. The rate
 * block carries `covered_steps` so a rate near the end of a run is not read as
 * covering a full bin when it does not.
 */
export interface GraphPopulationResult {
  id: string;
  label: string;
  model: string;
  count: number;
  neuron_type: StudioNeuronType;
  offset: number;
  n_spikes: number;
  mean_rate_hz: number;
  events: { step: number[]; neuron: number[] };
  rate: { bin_steps: number; bin_ms: number; time_ms: number[]; rate_hz: number[]; covered_steps: number };
}

/**
 * The connectivity a projection actually produced.
 *
 * `csr_sha256` identifies the matrix whether or not `indptr` and `indices` were
 * sent, so a run whose connectivity was too large to return is still citable.
 * `autapses_removed` is reported rather than silently applied, because a rule
 * that removed self-connections produced a different network from the one the
 * parameters describe.
 */
export interface GraphProjectionTopology {
  id: string;
  source: string;
  target: string;
  rule: ProjectionRule;
  seed: number;
  weight: number;
  delay_steps: number;
  delay_mode: string;
  n_synapses: number;
  autapses_removed: number;
  csr_sha256: string;
  indptr?: number[];
  indices?: number[];
}

/**
 * How the run was executed, in the detail a reproduction needs.
 *
 * Delay and synapse semantics, step order and projection latency are stated
 * because two implementations can agree on every parameter and disagree by a
 * timestep on when a spike arrives; that difference is invisible in the
 * numbers and decisive in the result.
 */
export interface GraphExecutionBlock {
  backend: { selected: string; rejected: { name: string; reason: string }[] };
  loop: string;
  step_order: string;
  projection_latency_steps: number;
  delay_semantics: string;
  synapse_semantics: string;
  network_dt_s: number;
  population_construction: string;
  autapses: string;
  state_check: string;
  runtime: { package_version: string; python: string; numpy: string };
}

/**
 * A graph run, or the refusal that stopped it.
 *
 * Everything but `success` is optional because a refusal carries `errors` and
 * nothing else. `topology.connectivity_included` and its reason exist so an
 * omitted connectivity block is read as omitted rather than as empty.
 */
export interface GraphSimResult {
  success: boolean;
  errors?: string[];
  schema_version?: string;
  spec?: Record<string, unknown> & { graph_sha256?: string; n_steps?: number; dt?: number };
  execution?: GraphExecutionBlock;
  populations?: GraphPopulationResult[];
  topology?: {
    n_synapses: number;
    connectivity_included: boolean;
    connectivity_omitted_reason: string | null;
    projections: GraphProjectionTopology[];
  };
  n_total?: number;
  n_spikes?: number;
  spike_times?: number[];
  spike_neurons?: number[];
  duration?: number;
  dt?: number;
  n_steps?: number;
  graph_summary?: {
    n_populations: number;
    n_projections: number;
    n_neurons: number;
    n_synapses: number;
    n_excitatory: number;
    n_inhibitory: number;
  };
  contract?: MetricContract;
}

/**
 * A NIR interchange document, carried opaquely.
 *
 * The nodes and edges are not narrowed here: NIR is another project's format,
 * and a type that lagged behind it would refuse documents this Studio should
 * pass through unchanged.
 */
export interface NIRFormat {
  format: string;
  version: string;
  nodes: Record<string, unknown>;
  edges: unknown[];
}

/** One stored workspace, as it appears in a list. */
export interface ProjectSummary {
  name: string;
  /** Null when the workspace's revisions cannot be read; it is still listed. */
  saved_at: number | null;
  revision: number | null;
  revision_count: number;
  version: string;
}

/** One workspace waiting in the recoverable trash. */
export interface DeletedProjectSummary {
  deleted_at: number | null;
  name: string;
  token: string;
}

/**
 * What a save wrote.
 *
 * `parent_revision` and `revision` together make the history a chain rather
 * than a sequence of numbers, so a fork or a lost update is visible after the
 * fact.
 */
export interface ProjectSaveResponse {
  evidence_classification: "project_workspace";
  name: string;
  /** The revision this save descends from; null for the first one. */
  parent_revision: number | null;
  project_sha256: string;
  /** The immutable revision this save wrote. */
  revision: number;
  saved_at: number;
  schema_version: "studio.project-save.v1";
  state_sha256: string;
  version: string;
}

/** One immutable revision as reported by `/api/project/{name}/revisions`. */
export interface ProjectRevision {
  name: string;
  parent: number | null;
  revision: number;
  saved_at: number;
  schema_version: string;
  state_sha256: string;
}

/** Every stored revision of one workspace. */
export interface ProjectRevisionList {
  name: string;
  revisions: ProjectRevision[];
}

/** What a refused edit became after being kept as its own branch. */
export interface ProjectBranchResponse {
  /** The workspace now holding the refused edit. */
  branched: string;
  /** The workspace it diverged from. */
  from: string;
  /** The revision it diverged at. */
  base_revision: number;
  /** The branch's own first revision, always 1. */
  revision: number;
}

/** The 409 body of a save made from a revision that is no longer current. */
export interface WorkspaceConflictDetail {
  actual_revision: number;
  error: "workspace_conflict";
  expected_revision: number | null;
  reason: string;
}

/** The 503 body of a save that could not take the workspace from another writer. */
export interface WorkspaceBusyDetail {
  error: "workspace_busy";
  name: string;
  reason: string;
  timeout_seconds: number;
}

/**
 * How far a pipeline run got, and where it stopped if it did.
 *
 * `step` names the stage that failed; `steps` carries what each stage
 * produced, so a failure at the end does not discard what the earlier ones
 * did.
 */
export interface PipelineResult {
  success: boolean;
  target: string;
  step?: string;
  errors?: string[];
  error?: string;
  steps?: Record<string, unknown>;
  pipeline?: string;
}

/**
 * One frame on the progress socket.
 *
 * `heartbeat` is a type of its own so a quiet operation is distinguishable
 * from a dropped connection, and `result` is `unknown` because it carries
 * whatever the completed operation returns.
 */
export interface ProgressMessage {
  type: "progress" | "complete" | "error" | "heartbeat";
  step?: string;
  pct?: number;
  msg?: string;
  result?: unknown;
}

/**
 * What `/api/codegen` returns: code that runs the same effective experiment.
 *
 * `request` is the pinned request the script carries — a drawn stochastic seed
 * is already fixed in it — and `experiment_sha256` is the digest the script
 * checks before it reports a result.
 */
export interface CodegenResponse {
  script: string;
  oneliner: string;
  replay_script: string;
  experiment_sha256: string;
  request: Record<string, unknown>;
}

/**
 * A sealed `studio.replay-pack.v1` document from `/api/export/replay-pack`.
 *
 * The fields the UI needs are named; the specification, expectation and
 * environment blocks travel opaquely to the file the user saves, because their
 * contract belongs to the Python replay runner, not to the browser.
 */
export interface ReplayPack {
  schema_version: "studio.replay-pack.v1";
  source: "model" | "ode";
  request: Record<string, unknown>;
  experiment: Record<string, unknown>;
  experiment_sha256: string;
  experiment_identity_sha256: string;
  expectation: Record<string, unknown>;
  environment: Record<string, string>;
  runner: { module: string; command: string; entrypoint: string };
}
