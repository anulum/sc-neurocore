// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Reproducible authored equation compiler settings

import type {
  EquationDetection, EquationExpression, EquationFunctionLowering, EquationIntegrationMethod,
  EquationIRRequest, EquationOverflow, EquationRandomPrecision, EquationRounding,
  EquationThresholdRng, NamedEquationValue,
} from "./api/equationIrTypes";

/** Integration choices understood by the public model constructor. */
export const EQUATION_METHODS = ["euler", "rk4", "exp_euler", "map", "gauss_seidel"] as const;
/** Event-detection modes; unsupported combinations remain explicit compiler refusals. */
export const EQUATION_DETECTIONS = ["level", "crossing", "poisson", "escape_rate"] as const;
/** Native integrated-state overflow choices. */
export const EQUATION_OVERFLOWS = ["saturate", "wrap", "trap"] as const;
/** Native product rounding choices. */
export const EQUATION_ROUNDINGS = ["truncate", "nearest", "bankers"] as const;
/** Existing lowering profile identities, without a fixed function-count promise. */
export const EQUATION_FUNCTION_PROFILES = ["lut-v1", "v2"] as const;
/** Existing normal-draw precision choices. */
export const EQUATION_RANDOM_PROFILES = ["standard", "exact"] as const;
/** Maintained stochastic-threshold generators. */
export const EQUATION_THRESHOLD_GENERATORS = ["lfsr16", "philox"] as const;
/** Explicit dimensional-validation modes. */
export const EQUATION_UNIT_MODES = ["none", "strict"] as const;

/** Editable arithmetic drafts retain incomplete numbers instead of substituting defaults. */
export interface EquationIrProfileDraft {
  module_name: string;
  data_width: string;
  fraction: string;
  signed: boolean;
  overflow: EquationOverflow;
  rounding: EquationRounding;
  pipeline_stages: string;
  pipeline_points: string;
  function_lowering: EquationFunctionLowering;
  random_precision: EquationRandomPrecision;
}
/** Model settings supplement the equation playground's expressions, values and timestep. */
export interface EquationIrModelDraft {
  method: EquationIntegrationMethod;
  detection: EquationDetection;
  substeps: string;
  rate_expression: string;
  probability_expression: string;
  spike_value: string;
  threshold_rng: EquationThresholdRng;
  rng_seed: string;
  random_stream: string;
  /** Comma-separated playground parameter names to freeze as immutable constants. */
  constant_names: string;
  units: "none" | "strict";
  /** Ordered JSON name/unit declarations include playground parameters frozen as constants. */
  state_units: string;
  parameter_units: string;
  dt_unit: string;
  input_unit: string;
  input_magnitude: string;
}
/** Complete persisted compiler drafts; missing legacy configuration alone gets defaults. */
export interface EquationIrConfiguration {
  schema_version: "studio.equation-ir-settings.v1";
  profile: EquationIrProfileDraft;
  model: EquationIrModelDraft;
}
/** Authored inputs already held by the equation playground. */
export interface EquationIrSourceInput {
  equations: string[];
  threshold: string;
  reset: string;
  odeParams: Record<string, number>;
  odeInit: Record<string, number>;
  dt: number;
  equationIrConfig: EquationIrConfiguration;
}

/**
 * Create independent defaults matching the native constructor and numeric profile.
 *
 * @returns Fresh compiler drafts, including the documented LFSR default key.
 */
export function defaultEquationIrConfiguration(): EquationIrConfiguration {
  return {
    schema_version: "studio.equation-ir-settings.v1",
    profile: {
      module_name: "sc_equation_neuron", data_width: "16", fraction: "8", signed: true,
      overflow: "saturate", rounding: "truncate", pipeline_stages: "0", pipeline_points: "",
      function_lowering: "lut-v1", random_precision: "standard",
    },
    model: {
      method: "euler", detection: "level", substeps: "1", rate_expression: "",
      probability_expression: "", spike_value: "", threshold_rng: "lfsr16",
      rng_seed: "44257", random_stream: "0", constant_names: "", units: "none",
      state_units: "[]", parameter_units: "[]", dt_unit: "", input_unit: "", input_magnitude: "1",
    },
  };
}

/**
 * Read one explicit enumerated option without casting or selecting a substitute.
 *
 * @param value - Stored or edited option.
 * @param choices - Maintained choices.
 * @param label - Caller-facing field name.
 * @returns The matching literal.
 * @throws {Error} When no maintained choice matches.
 */
export function equationIrChoice<T extends string>(value: unknown, choices: readonly T[], label: string): T {
  const choice = choices.find((item) => item === value);
  if (choice === undefined) throw new Error(`${label} has an unsupported value.`);
  return choice;
}

/**
 * Restore a complete saved draft, refusing explicit malformed settings.
 *
 * Numeric text may be incomplete: it remains visible and cannot compile until
 * corrected. Its exact text is preserved when projects or sessions are saved.
 *
 * @param value - Saved configuration, absent in legacy projects.
 * @returns Complete independent settings.
 * @throws {Error} When an explicit document cannot be restored losslessly.
 */
export function readEquationIrConfiguration(value: unknown): EquationIrConfiguration {
  if (value === undefined) return defaultEquationIrConfiguration();
  const root = record(value, ["schema_version", "profile", "model"], "Compiler settings");
  if (root.schema_version !== "studio.equation-ir-settings.v1") {
    throw new Error("Compiler settings have an unsupported schema.");
  }
  const profile = record(root.profile, [
    "module_name", "data_width", "fraction", "signed", "overflow", "rounding",
    "pipeline_stages", "pipeline_points", "function_lowering", "random_precision",
  ], "Compiler profile");
  const model = record(root.model, [
    "method", "detection", "substeps", "rate_expression", "probability_expression", "spike_value",
    "threshold_rng", "rng_seed", "random_stream", "constant_names", "units", "state_units",
    "parameter_units", "dt_unit", "input_unit", "input_magnitude",
  ], "Compiler model settings");
  if (typeof profile.signed !== "boolean") throw new Error("Compiler signedness must be Boolean.");
  return {
    schema_version: "studio.equation-ir-settings.v1",
    profile: {
      module_name: text(profile.module_name, "Module name"),
      data_width: text(profile.data_width, "Total bits"),
      fraction: text(profile.fraction, "Fractional bits"), signed: profile.signed,
      overflow: equationIrChoice(profile.overflow, EQUATION_OVERFLOWS, "Overflow"),
      rounding: equationIrChoice(profile.rounding, EQUATION_ROUNDINGS, "Rounding"),
      pipeline_stages: text(profile.pipeline_stages, "Product pipeline stages"),
      pipeline_points: text(profile.pipeline_points, "Product pipeline points"),
      function_lowering: equationIrChoice(profile.function_lowering, EQUATION_FUNCTION_PROFILES, "Function profile"),
      random_precision: equationIrChoice(profile.random_precision, EQUATION_RANDOM_PROFILES, "Random precision"),
    },
    model: {
      method: equationIrChoice(model.method, EQUATION_METHODS, "Integrator"),
      detection: equationIrChoice(model.detection, EQUATION_DETECTIONS, "Event detection"),
      substeps: text(model.substeps, "Substeps"),
      rate_expression: text(model.rate_expression, "Rate expression"),
      probability_expression: text(model.probability_expression, "Probability expression"),
      spike_value: text(model.spike_value, "Graded spike expression"),
      threshold_rng: equationIrChoice(model.threshold_rng, EQUATION_THRESHOLD_GENERATORS, "Threshold generator"),
      rng_seed: text(model.rng_seed, "Compiler seed"), random_stream: text(model.random_stream, "Random stream"),
      constant_names: text(model.constant_names, "Immutable constants"),
      units: equationIrChoice(model.units, EQUATION_UNIT_MODES, "Units mode"),
      state_units: text(model.state_units, "State units"),
      parameter_units: text(model.parameter_units, "Parameter units"),
      dt_unit: text(model.dt_unit, "Timestep unit"),
      input_unit: text(model.input_unit, "Drive unit"),
      input_magnitude: text(model.input_magnitude, "Drive unit magnitude"),
    },
  };
}

/**
 * Translate playground source and saved drafts to the exact versioned request.
 *
 * Declaration arrays preserve author order through sorted worker JSON. Constant
 * selections deliberately freeze named playground parameter values. Unknown unit
 * keys and constant selections refuse instead of disappearing. Initial-state
 * names also declare held words; missing numeric equation states stay implicit.
 *
 * @param input - Current equation playground and compiler configuration.
 * @returns Full native compiler inputs, with exact seed text.
 * @throws {Error} When a draft is incomplete or would lose authored information.
 */
export function equationIrRequest(input: EquationIrSourceInput): EquationIRRequest {
  const config = readEquationIrConfiguration(input.equationIrConfig);
  const p = config.profile, m = config.model;
  const equations = input.equations.map((line) => declaration(line, true));
  if (equations.length === 0) throw new Error("Add an equation before compiling.");
  unique(equations.map((item) => item.name), "Equations");
  const states = equations.map((item) => item.name);
  const constants = names(m.constant_names, "Immutable constants");
  const points = names(p.pipeline_points, "Product pipeline points");
  declared(constants, Object.keys(input.odeParams), "Immutable constants");
  const initialStates = Object.keys(input.odeInit);
  const declaredStates = [...states, ...initialStates];
  const stateUnits = unitMap(m.state_units, declaredStates, "State units");
  declared([...stateUnits.keys()], Object.keys(input.odeInit), "Initial state units");
  if (m.units === "strict" && states.some((name) => !initialStates.includes(name))) {
    throw new Error("Strict units require an initial value for every equation state.");
  }
  const parameterUnits = unitMap(m.parameter_units, Object.keys(input.odeParams), "Parameter units");
  const dtUnit = optional(m.dt_unit), inputUnit = optional(m.input_unit);
  const dt = finite(input.dt, "Timestep");
  if (dt <= 0) throw new Error("Timestep must be positive.");
  if (m.units === "none" && (stateUnits.size > 0
    || parameterUnits.size > 0 || dtUnit !== null || inputUnit !== null)) {
    throw new Error("Physical unit annotations require strict units mode.");
  }
  if (!/^(0|[1-9][0-9]*)$/.test(m.rng_seed) || m.rng_seed.length > 20
    || BigInt(m.rng_seed) >= 2n ** 64n) {
    throw new Error("Compiler seed must be an unsigned 64-bit decimal string.");
  }
  const stream = integer(m.random_stream, 0, "Random stream");
  if (stream >= 2 ** 32) throw new Error("Random stream must be below 2^32.");
  if (integer(p.pipeline_stages, 0, "Product pipeline stages") > 0 && points.length > 0) {
    throw new Error("Choose global product pipeline stages or explicit product points.");
  }
  const parameters = Object.entries(input.odeParams)
    .filter(([name]) => !constants.includes(name))
    .map(([name, value]) => namedValue(name, value, parameterUnits));
  const reset = input.reset.split(/[;\n]/).filter((line) => line.trim().length > 0)
    .map((line) => declaration(line, false));
  unique(reset.map((item) => item.name), "Reset");
  declared(reset.map((item) => item.name), declaredStates, "Reset");
  return {
    schema_version: "studio.equation-ir-request.v1",
    model: {
      equations, initial_state: Object.entries(input.odeInit)
        .map(([name, value]) => namedValue(name, value, stateUnits)),
      parameters, constants: constants.map((name) => namedValue(name, input.odeParams[name], parameterUnits)),
      reset, threshold: optional(input.threshold),
      dt: { value: dt, unit: dtUnit },
      method: m.method, detection: m.detection, substeps: integer(m.substeps, 1, "Substeps"),
      rate_expression: optional(m.rate_expression), probability_expression: optional(m.probability_expression),
      spike_value: optional(m.spike_value), threshold_rng: m.threshold_rng,
      rng_seed: m.rng_seed, random_stream: stream, units: m.units,
      input_unit: inputUnit === null ? null
        : { value: magnitude(m.input_magnitude, "Drive unit magnitude"), unit: inputUnit },
    },
    profile: {
      module_name: p.module_name, data_width: integer(p.data_width, 1, "Total bits"),
      fraction: integer(p.fraction, 0, "Fractional bits"), signed: p.signed,
      overflow: p.overflow, rounding: p.rounding,
      pipeline_stages: integer(p.pipeline_stages, 0, "Product pipeline stages"),
      pipeline_points: points, function_lowering: p.function_lowering, random_precision: p.random_precision,
    },
  };
}

/**
 * Read a closed saved object without dropping unknown fields.
 *
 * @param value - Unknown saved value.
 * @param keys - Exact field set.
 * @param label - Field name for a refusal.
 * @returns The original validated record.
 * @throws {Error} When the object shape differs.
 */
function record(value: unknown, keys: string[], label: string): Record<string, unknown> {
  if (!isRecord(value)
    || Object.keys(value).length !== keys.length || Object.keys(value).some((key) => !keys.includes(key))) {
    throw new Error(`${label} cannot be restored without changing its fields.`);
  }
  return value;
}

/**
 * Narrow an actual JSON object before inspecting untrusted saved members.
 *
 * @param value - Candidate object.
 * @returns Whether named unknown members can be read safely.
 */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Read one saved text draft without coercion.
 *
 * @param value - Saved value.
 * @param label - Field name.
 * @returns Exact text.
 * @throws {Error} When the field is not text.
 */
function text(value: unknown, label: string): string {
  if (typeof value !== "string") throw new Error(`${label} must be text.`);
  return value;
}

/**
 * Read canonical unsigned integer text into an exact safe JS integer.
 *
 * @param value - User draft.
 * @param minimum - Smallest admitted value.
 * @param label - Field name.
 * @returns The exact integer.
 * @throws {Error} When the text or integer domain is invalid.
 */
function integer(value: string, minimum: number, label: string): number {
  const parsed = Number(value);
  if (!/^(0|[1-9][0-9]*)$/.test(value) || !Number.isSafeInteger(parsed) || parsed < minimum) {
    throw new Error(`${label} must be an exact integer of at least ${minimum}.`);
  }
  return parsed;
}

/**
 * Require a finite existing playground value.
 *
 * @param value - Current numeric value.
 * @param label - Field name.
 * @returns The value unchanged.
 * @throws {Error} When no finite scalar is present.
 */
function finite(value: number | undefined, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${label} must be finite.`);
  return value;
}

/**
 * Parse an explicit finite decimal quantity magnitude.
 *
 * @param value - User text.
 * @param label - Field name.
 * @returns Numeric magnitude.
 * @throws {Error} When the text is not a finite decimal number.
 */
function magnitude(value: string, label: string): number {
  if (!/^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$/.test(value)) {
    throw new Error(`${label} must be a finite decimal number.`);
  }
  return finite(Number(value), label);
}

/**
 * Preserve authored nonblank expressions and unit text.
 *
 * @param value - Draft text.
 * @returns Original text or explicit absence.
 */
function optional(value: string): string | null {
  return value.trim().length === 0 ? null : value;
}

/**
 * Reject repeated declarations before dictionary construction.
 *
 * @param values - Ordered names.
 * @param label - Declaration category.
 * @throws {Error} When a name repeats.
 */
function unique(values: string[], label: string): void {
  if (new Set(values).size !== values.length) throw new Error(`${label} cannot repeat a name.`);
}

/**
 * Require every selected or annotated name to exist in the authored source.
 *
 * @param values - Referenced names.
 * @param allowed - Declared names.
 * @param label - Declaration category.
 * @throws {Error} When an unknown name would otherwise disappear.
 */
function declared(values: string[], allowed: string[], label: string): void {
  const missing = values.find((name) => !allowed.includes(name));
  if (missing !== undefined) throw new Error(`${label} names undeclared symbol ${missing}.`);
}

/**
 * Read a comma-separated selection, retaining order and refusing empty members.
 *
 * @param value - Saved user draft.
 * @param label - Selection name.
 * @returns Ordered unique names, or an explicit empty selection.
 * @throws {Error} When a member is blank or repeated.
 */
function names(value: string, label: string): string[] {
  if (value.trim().length === 0) return [];
  const selected = value.split(",").map((name) => name.trim());
  if (selected.some((name) => name.length === 0)) throw new Error(`${label} contains an empty name.`);
  unique(selected, label);
  return selected;
}

/**
 * Read and validate all physical unit annotations without filtering any key.
 *
 * @param value - JSON draft.
 * @param allowed - Declared names.
 * @param label - Annotation category.
 * @returns Exact unit strings indexed by unique declared names.
 * @throws {Error} When syntax, types or declaration names are invalid.
 */
function unitMap(value: string, allowed: string[], label: string): Map<string, string> {
  let parsed: unknown;
  try { parsed = JSON.parse(value); }
  catch (cause: unknown) { throw new Error(`${label} must be an ordered JSON array of name/unit declarations.`, { cause }); }
  if (!Array.isArray(parsed)) {
    throw new Error(`${label} must be an ordered JSON array of name/unit declarations.`);
  }
  const units = new Map<string, string>();
  for (const item of parsed) {
    const annotation = record(item, ["name", "unit"], label);
    const name = text(annotation.name, label), unit = text(annotation.unit, label);
    declared([name], allowed, label);
    if (unit.trim().length === 0) throw new Error(`${label} requires nonblank unit strings.`);
    if (units.has(name)) throw new Error(`${label} cannot repeat a name.`);
    units.set(name, unit);
  }
  return units;
}

/**
 * Translate one playground declaration while retaining declaration order.
 *
 * @param line - Derivative equation or reset assignment.
 * @param derivative - Whether the left side must be a time derivative.
 * @returns State name and nonempty expression.
 * @throws {Error} When the declaration cannot be translated faithfully.
 */
function declaration(line: string, derivative: boolean): EquationExpression {
  const match = derivative
    ? /^\s*d([A-Za-z_][A-Za-z_0-9]*)\s*\/\s*dt\s*=([\s\S]+)$/.exec(line)
    : /^\s*([A-Za-z_][A-Za-z_0-9]*)\s*=([\s\S]+)$/.exec(line);
  const name = match?.[1], expression = match?.[2]?.trim();
  if (name === undefined || expression === undefined || expression.length === 0) {
    throw new Error(derivative ? "Use dstate/dt = expression for each equation." : "Use state = expression for each reset.");
  }
  return { name, expression };
}

/**
 * Bind one scalar to its exact authored symbol and optional physical unit.
 *
 * @param name - Declared name.
 * @param value - Finite magnitude.
 * @param units - Validated annotations.
 * @returns Full ordered scalar declaration.
 * @throws {Error} When the magnitude is not finite.
 */
function namedValue(name: string, value: number | undefined, units: Map<string, string>): NamedEquationValue {
  return { name, value: finite(value, name), unit: units.get(name) ?? null };
}
