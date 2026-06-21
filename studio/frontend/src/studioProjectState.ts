// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio project state snapshot helpers

import type { PopulationNode, ProjectionEdge } from "./api/client";
import type { StudioSimulationSourceMode } from "./studioSimulationConfig";

export interface StudioProjectTrainingConfig {
  dataset: string;
  epochs: number;
  batch_size: number;
  lr: number;
  hidden: number[];
  timesteps: number;
  surrogate: string;
  learn_beta: boolean;
  learn_threshold: boolean;
}

export interface StudioProjectStateSnapshot extends Record<string, unknown> {
  sourceMode: StudioSimulationSourceMode;
  equations: string[];
  threshold: string;
  reset: string;
  odeParams: Record<string, number>;
  odeInit: Record<string, number>;
  selectedModelName: string;
  modelParams: Record<string, number>;
  dt: number;
  duration: number;
  current: number;
  protocol: string;
  graphPopulations: PopulationNode[];
  graphProjections: ProjectionEdge[];
  synthTarget: string;
  trainingConfig: StudioProjectTrainingConfig;
}

export function studioProjectSaveState(input: StudioProjectStateSnapshot): StudioProjectStateSnapshot {
  return {
    sourceMode: input.sourceMode,
    equations: input.equations,
    threshold: input.threshold,
    reset: input.reset,
    odeParams: input.odeParams,
    odeInit: input.odeInit,
    selectedModelName: input.selectedModelName,
    modelParams: input.modelParams,
    dt: input.dt,
    duration: input.duration,
    current: input.current,
    protocol: input.protocol,
    graphPopulations: input.graphPopulations,
    graphProjections: input.graphProjections,
    synthTarget: input.synthTarget,
    trainingConfig: input.trainingConfig,
  };
}

export function studioProjectStateFromLoadResponse(
  response: unknown,
  fallbackTrainingConfig: StudioProjectTrainingConfig,
): StudioProjectStateSnapshot {
  const state = recordValue(recordValue(response).state);
  return {
    sourceMode: sourceModeValue(state.sourceMode),
    equations: stringArrayValue(state.equations),
    threshold: stringValue(state.threshold, ""),
    reset: stringValue(state.reset, ""),
    odeParams: numberRecordValue(state.odeParams),
    odeInit: numberRecordValue(state.odeInit),
    selectedModelName: stringValue(state.selectedModelName, ""),
    modelParams: numberRecordValue(state.modelParams),
    dt: positiveNumberValue(state.dt, 0.1),
    duration: positiveNumberValue(state.duration, 100),
    current: finiteNumberValue(state.current, 10),
    protocol: stringValue(state.protocol, "constant"),
    graphPopulations: populationArrayValue(state.graphPopulations),
    graphProjections: projectionArrayValue(state.graphProjections),
    synthTarget: stringValue(state.synthTarget, "ice40"),
    trainingConfig: trainingConfigValue(state.trainingConfig, fallbackTrainingConfig),
  };
}

function recordValue(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function sourceModeValue(value: unknown): StudioSimulationSourceMode {
  return value === "ode" ? "ode" : "model";
}

function stringValue(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}

function finiteNumberValue(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function positiveNumberValue(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : fallback;
}

function booleanValue(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function stringArrayValue(value: unknown): string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string") ? value : [];
}

function numberRecordValue(value: unknown): Record<string, number> {
  const record = recordValue(value);
  return Object.fromEntries(
    Object.entries(record).filter((entry): entry is [string, number] =>
      typeof entry[1] === "number" && Number.isFinite(entry[1])),
  );
}

function numberArrayValue(value: unknown, fallback: number[]): number[] {
  return Array.isArray(value) && value.every((item) => typeof item === "number" && Number.isFinite(item))
    ? value
    : fallback;
}

function populationArrayValue(value: unknown): PopulationNode[] {
  return Array.isArray(value) ? value.filter(isRecord) as unknown as PopulationNode[] : [];
}

function projectionArrayValue(value: unknown): ProjectionEdge[] {
  return Array.isArray(value) ? value.filter(isRecord) as unknown as ProjectionEdge[] : [];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function trainingConfigValue(
  value: unknown,
  fallback: StudioProjectTrainingConfig,
): StudioProjectTrainingConfig {
  const config = recordValue(value);
  return {
    dataset: stringValue(config.dataset, fallback.dataset),
    epochs: finiteNumberValue(config.epochs, fallback.epochs),
    batch_size: finiteNumberValue(config.batch_size, fallback.batch_size),
    lr: finiteNumberValue(config.lr, fallback.lr),
    hidden: numberArrayValue(config.hidden, fallback.hidden),
    timesteps: finiteNumberValue(config.timesteps, fallback.timesteps),
    surrogate: stringValue(config.surrogate, fallback.surrogate),
    learn_beta: booleanValue(config.learn_beta, fallback.learn_beta),
    learn_threshold: booleanValue(config.learn_threshold, fallback.learn_threshold),
  };
}
