// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio model selection store state helpers

/**
 * The state patches that follow from choosing what to run.
 *
 * Selecting a model or a preset replaces most of the experiment, so these are
 * the largest patches in the store. A preset is read defensively — it is a
 * stored document that may predate any field it is missing — which is why
 * `presetSelection` reads each value through a typed accessor with a fallback
 * rather than trusting the shape.
 */
import type { ModelDetail, ModelSummary, NeuronTemplate, PresetSummary } from "./api/client";
import type { StudioSimulationSourceMode } from "./studioSimulationConfig";

/** The tabs a selection may move the Studio to. */
export type StudioSelectionViewTab =
  "trace" | "phase" | "isi" | "fi-curve" | "bifurcation" | "sensitivity" |
  "precision" | "heatmap" | "verilog" | "code" | "compare" | "freq" |
  "sta" | "characterize" | "multi" | "network" | "ir" | "synth" |
  "train" | "canvas" | "admin";

/** What a preset asks the Studio to do once it is loaded. */
export type StudioPresetPostLoadAction =
  | { kind: "fi-curve" }
  | { kind: "precision" }
  | { activeTab: StudioSelectionViewTab; kind: "simulate" };

/** The editable templates arrived. */
export interface TemplatesLoadedStatePatch {
  templates: NeuronTemplate[];
}

/** The catalogue arrived. */
export interface ModelsLoadedStatePatch {
  models: ModelSummary[];
}

/** The saved presets arrived. */
export interface PresetsLoadedStatePatch {
  presets: PresetSummary[];
}

/** A template was chosen: its equations, parameters and protocol replace the experiment. */
export interface TemplateSelectedStatePatch {
  current: number;
  dt: number;
  duration: number;
  equations: string[];
  error: null;
  fiResult: null;
  odeInit: Record<string, number>;
  odeParams: Record<string, number>;
  reset: string;
  result: null;
  sourceMode: StudioSimulationSourceMode;
  threshold: string;
}

/** A model was chosen and its contract is being fetched. */
export interface ModelSelectionStartedStatePatch {
  error: null;
  fiResult: null;
  result: null;
  selectedModelName: string;
}

/** A model's contract arrived: its defaults become the experiment. */
export interface ModelDetailLoadedStatePatch {
  dt: number;
  modelDetail: ModelDetail;
  modelParams: Record<string, number>;
  modelIntegrator: string;
  modelQFormat: string;
  sourceMode: "model";
}

/** The run settings a preset carries, as against the model it names. */
export interface ModelPresetRuntimeStatePatch {
  current: number;
  duration: number;
  protocol: string;
}

/** The typed-equation experiment a preset carries. */
export interface OdePresetSelectedStatePatch {
  current: number;
  dt: number;
  duration: number;
  equations: string[];
  odeInit: Record<string, number>;
  odeParams: Record<string, number>;
  protocol: string;
  reset: string;
  sourceMode: "ode";
  threshold: string;
}

/** Everything a preset asks for: which model, which run settings, and what to do next. */
export interface StudioPresetSelection {
  action: StudioPresetPostLoadAction;
  modelName: string | null;
  modelRuntimeState: ModelPresetRuntimeStatePatch | null;
  odeState: OdePresetSelectedStatePatch | null;
}

/**
 * Take the templates into the store.
 *
 * @param templates - The templates.
 * @returns The patch.
 */
export function templatesLoadedState(templates: NeuronTemplate[]): TemplatesLoadedStatePatch {
  return { templates };
}

/**
 * Take the catalogue into the store.
 *
 * @param models - The catalogue.
 * @returns The patch.
 */
export function modelsLoadedState(models: ModelSummary[]): ModelsLoadedStatePatch {
  return { models };
}

/**
 * Take the presets into the store.
 *
 * @param presets - The presets.
 * @returns The patch.
 */
export function presetsLoadedState(presets: PresetSummary[]): PresetsLoadedStatePatch {
  return { presets };
}

/**
 * Replace the experiment with a template's own definition.
 *
 * @param template - The chosen template.
 * @returns The patch.
 */
export function templateSelectedState(template: NeuronTemplate): TemplateSelectedStatePatch {
  return {
    current: template.current,
    dt: template.dt,
    duration: template.duration,
    equations: [...template.equations],
    error: null,
    fiResult: null,
    odeInit: { ...template.init },
    odeParams: { ...template.params },
    reset: template.reset,
    result: null,
    sourceMode: "ode",
    threshold: template.threshold,
  };
}

/**
 * Record that a model was chosen, before its contract has arrived.
 *
 * @param selectedModelName - The model's name.
 * @returns The patch.
 */
export function modelSelectionStartedState(
  selectedModelName: string,
): ModelSelectionStartedStatePatch {
  return { error: null, fiResult: null, result: null, selectedModelName };
}

/**
 * Replace the experiment with a model's own defaults.
 *
 * @param detail - The model's contract.
 * @returns The patch.
 */
export function modelDetailLoadedState(detail: ModelDetail): ModelDetailLoadedStatePatch {
  const compileConfiguration = detail.compile_configuration;
  return {
    dt: detail.dt,
    modelDetail: detail,
    modelParams: modelDefaultParameters(detail),
    modelIntegrator: compileConfiguration?.default_integrator ?? detail.integration_method,
    modelQFormat: compileConfiguration?.default_q_format ?? "Q8.8",
    sourceMode: "model",
  };
}

/**
 * A model's parameters and initial state, as one record of numbers.
 *
 * The two are flattened together because the Studio edits them in one place;
 * the model's contract is what distinguishes them when it matters.
 *
 * @param detail - The model's contract.
 * @returns Name to default value.
 */
export function modelDefaultParameters(detail: ModelDetail): Record<string, number> {
  const modelParams: Record<string, number> = {};
  for (const parameter of detail.params) modelParams[parameter.name] = parameter.default;
  for (const stateVariable of detail.state_vars) {
    modelParams[stateVariable.name] = stateVariable.default;
  }
  return modelParams;
}

/**
 * Read a stored preset into everything it asks the Studio to do.
 *
 * The preset is a stored document that may predate any field it is missing, so
 * every value is read through a typed accessor with a fallback rather than
 * being trusted.
 *
 * @param preset - The stored preset.
 * @returns What it selects and what to do next.
 */
export function presetSelection(preset: Record<string, unknown>): StudioPresetSelection {
  return {
    action: presetPostLoadAction(preset.suggested_view),
    modelName: modelPresetName(preset),
    modelRuntimeState: modelPresetRuntimeState(preset),
    odeState: odePresetState(preset),
  };
}

/**
 * The model a preset names, when it names one at all.
 *
 * @param preset - The stored preset.
 * @returns The model's name, or `null` for a preset that is not model-based.
 */
function modelPresetName(preset: Record<string, unknown>): string | null {
  return preset.mode === "model" && typeof preset.model_name === "string"
    ? preset.model_name
    : null;
}

/**
 * The run settings a preset carries, with the Studio's defaults for what it omits.
 *
 * @param preset - The stored preset.
 * @returns The settings.
 */
function modelPresetRuntimeState(preset: Record<string, unknown>): ModelPresetRuntimeStatePatch {
  return {
    current: finiteNumberValue(preset.current, 10),
    duration: finiteNumberValue(preset.duration, 200),
    protocol: stringValue(preset.protocol, "constant"),
  };
}

/**
 * The typed-equation experiment a preset carries, if it carries one.
 *
 * @param preset - The stored preset.
 * @returns The experiment, or `null` when the preset has no equations.
 */
function odePresetState(preset: Record<string, unknown>): OdePresetSelectedStatePatch | null {
  const equations = stringArrayValue(preset.equations);
  if (equations.length === 0) return null;
  return {
    current: finiteNumberValue(preset.current, 10),
    dt: finiteNumberValue(preset.dt, 0.1),
    duration: finiteNumberValue(preset.duration, 200),
    equations,
    odeInit: numberRecordValue(preset.init),
    odeParams: numberRecordValue(preset.params),
    protocol: stringValue(preset.protocol, "constant"),
    reset: stringValue(preset.reset, ""),
    sourceMode: "ode",
    threshold: stringValue(preset.threshold, ""),
  };
}

/**
 * What a preset asks the Studio to do once it is loaded.
 *
 * @param value - The preset's `suggested_view`, of any shape.
 * @returns The action, falling back to showing the trace.
 */
function presetPostLoadAction(value: unknown): StudioPresetPostLoadAction {
  if (value === "fi-curve") return { kind: "fi-curve" };
  if (value === "precision") return { kind: "precision" };
  return { activeTab: viewTabValue(value), kind: "simulate" };
}

/**
 * Read a stored tab name, ignoring one this build does not know.
 *
 * @param value - The stored value.
 * @returns The tab, or the trace tab.
 */
function viewTabValue(value: unknown): StudioSelectionViewTab {
  return typeof value === "string" && isSelectionViewTab(value) ? value : "trace";
}

/**
 * Whether a string is one of the tabs a preset may select.
 *
 * @param value - The string.
 * @returns Whether it is a tab.
 */
function isSelectionViewTab(value: string): value is StudioSelectionViewTab {
  return [
    "trace", "phase", "isi", "fi-curve", "bifurcation", "sensitivity",
    "precision", "heatmap", "verilog", "code", "compare", "freq", "sta",
    "characterize", "multi", "network", "ir", "synth", "train", "canvas",
    "admin",
  ].includes(value);
}

/**
 * Read a stored string, falling back when it is not one.
 *
 * @param value - The stored value.
 * @param fallback - What to use instead.
 * @returns The string.
 */
function stringValue(value: unknown, fallback: string): string {
  return typeof value === "string" && value.length > 0 ? value : fallback;
}

/**
 * Read a stored number, falling back for anything not finite.
 *
 * `NaN` and the infinities are rejected as well as non-numbers, because each
 * of them would travel silently into a request.
 *
 * @param value - The stored value.
 * @param fallback - What to use instead.
 * @returns The number.
 */
function finiteNumberValue(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

/**
 * Read a stored list of strings, dropping anything that is not one.
 *
 * @param value - The stored value.
 * @returns The strings, empty when there were none.
 */
function stringArrayValue(value: unknown): string[] {
  return Array.isArray(value) && value.every((item): item is string => typeof item === "string")
    ? [...value]
    : [];
}

/**
 * Read a stored record of numbers, dropping entries that are not.
 *
 * @param value - The stored value.
 * @returns The record, empty when there was none.
 */
function numberRecordValue(value: unknown): Record<string, number> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return {};
  const record: Record<string, number> = {};
  for (const [key, item] of Object.entries(value)) {
    if (typeof item === "number" && Number.isFinite(item)) record[key] = item;
  }
  return record;
}
