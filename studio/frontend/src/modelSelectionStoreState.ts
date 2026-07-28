// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio model selection store state helpers
import type { ModelDetail, ModelSummary, NeuronTemplate, PresetSummary } from "./api/client";
import type { StudioSimulationSourceMode } from "./studioSimulationConfig";

export type StudioSelectionViewTab =
  "trace" | "phase" | "isi" | "fi-curve" | "bifurcation" | "sensitivity" |
  "precision" | "heatmap" | "verilog" | "code" | "compare" | "freq" |
  "sta" | "characterize" | "multi" | "network" | "ir" | "synth" |
  "train" | "canvas" | "admin";

export type StudioPresetPostLoadAction =
  | { kind: "fi-curve" }
  | { kind: "precision" }
  | { activeTab: StudioSelectionViewTab; kind: "simulate" };

export interface TemplatesLoadedStatePatch {
  templates: NeuronTemplate[];
}

export interface ModelsLoadedStatePatch {
  models: ModelSummary[];
}

export interface PresetsLoadedStatePatch {
  presets: PresetSummary[];
}

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

export interface ModelSelectionStartedStatePatch {
  error: null;
  fiResult: null;
  result: null;
  selectedModelName: string;
}

export interface ModelDetailLoadedStatePatch {
  dt: number;
  modelDetail: ModelDetail;
  modelParams: Record<string, number>;
  modelIntegrator: string;
  modelQFormat: string;
  sourceMode: "model";
}

export interface ModelPresetRuntimeStatePatch {
  current: number;
  duration: number;
  protocol: string;
}

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

export interface StudioPresetSelection {
  action: StudioPresetPostLoadAction;
  modelName: string | null;
  modelRuntimeState: ModelPresetRuntimeStatePatch | null;
  odeState: OdePresetSelectedStatePatch | null;
}

export function templatesLoadedState(templates: NeuronTemplate[]): TemplatesLoadedStatePatch {
  return { templates };
}

export function modelsLoadedState(models: ModelSummary[]): ModelsLoadedStatePatch {
  return { models };
}

export function presetsLoadedState(presets: PresetSummary[]): PresetsLoadedStatePatch {
  return { presets };
}

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

export function modelSelectionStartedState(
  selectedModelName: string,
): ModelSelectionStartedStatePatch {
  return { error: null, fiResult: null, result: null, selectedModelName };
}

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

export function modelDefaultParameters(detail: ModelDetail): Record<string, number> {
  const modelParams: Record<string, number> = {};
  for (const parameter of detail.params) modelParams[parameter.name] = parameter.default;
  for (const stateVariable of detail.state_vars) {
    modelParams[stateVariable.name] = stateVariable.default;
  }
  return modelParams;
}

export function presetSelection(preset: Record<string, unknown>): StudioPresetSelection {
  return {
    action: presetPostLoadAction(preset.suggested_view),
    modelName: modelPresetName(preset),
    modelRuntimeState: modelPresetRuntimeState(preset),
    odeState: odePresetState(preset),
  };
}

function modelPresetName(preset: Record<string, unknown>): string | null {
  return preset.mode === "model" && typeof preset.model_name === "string"
    ? preset.model_name
    : null;
}

function modelPresetRuntimeState(preset: Record<string, unknown>): ModelPresetRuntimeStatePatch {
  return {
    current: finiteNumberValue(preset.current, 10),
    duration: finiteNumberValue(preset.duration, 200),
    protocol: stringValue(preset.protocol, "constant"),
  };
}

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

function presetPostLoadAction(value: unknown): StudioPresetPostLoadAction {
  if (value === "fi-curve") return { kind: "fi-curve" };
  if (value === "precision") return { kind: "precision" };
  return { activeTab: viewTabValue(value), kind: "simulate" };
}

function viewTabValue(value: unknown): StudioSelectionViewTab {
  return typeof value === "string" && isSelectionViewTab(value) ? value : "trace";
}

function isSelectionViewTab(value: string): value is StudioSelectionViewTab {
  return [
    "trace", "phase", "isi", "fi-curve", "bifurcation", "sensitivity",
    "precision", "heatmap", "verilog", "code", "compare", "freq", "sta",
    "characterize", "multi", "network", "ir", "synth", "train", "canvas",
    "admin",
  ].includes(value);
}

function stringValue(value: unknown, fallback: string): string {
  return typeof value === "string" && value.length > 0 ? value : fallback;
}

function finiteNumberValue(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function stringArrayValue(value: unknown): string[] {
  return Array.isArray(value) && value.every((item): item is string => typeof item === "string")
    ? [...value]
    : [];
}

function numberRecordValue(value: unknown): Record<string, number> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return {};
  const record: Record<string, number> = {};
  for (const [key, item] of Object.entries(value)) {
    if (typeof item === "number" && Number.isFinite(item)) record[key] = item;
  }
  return record;
}
