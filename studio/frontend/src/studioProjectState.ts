// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio project state snapshot helpers

import { StudioRequestError } from "./api/client";
import type {
  DeletedProjectSummary,
  PopulationNode,
  ProjectSaveResponse,
  ProjectSummary,
  ProjectionEdge,
} from "./api/client";
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

export interface StudioProjectSnapshotInput {
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

export interface StudioProjectStateSnapshot
  extends StudioProjectSnapshotInput, Record<string, unknown> {}

/** Which workspace revision the editor is currently working from. */
export interface StudioProjectRevisionPointer {
  name: string;
  revision: number;
}

export interface StudioProjectSavedStatePatch {
  projectSaveResult: ProjectSaveResponse;
  projectRevision: StudioProjectRevisionPointer;
}

export interface StudioProjectRestorePointerPatch {
  projectRevision: StudioProjectRevisionPointer | null;
}

export interface StudioProjectListLoadedStatePatch {
  serverProjects: ProjectSummary[];
}

export interface StudioProjectDeletedListedStatePatch {
  deletedProjects: DeletedProjectSummary[];
}

export interface StudioProjectFailureStatePatch {
  error: string;
}

export function studioProjectSaveState(input: StudioProjectSnapshotInput): StudioProjectStateSnapshot {
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

export function studioProjectSavedState(
  projectSaveResult: ProjectSaveResponse,
): StudioProjectSavedStatePatch {
  return {
    projectRevision: {
      name: projectSaveResult.name,
      revision: projectSaveResult.revision,
    },
    projectSaveResult,
  };
}

/**
 * The revision to send as `expected_revision` when saving under `name`.
 *
 * Null means "the editor has not loaded this workspace", which the server
 * reads as a claim that it is new. Saving under a different name than the one
 * that was loaded is such a claim, so the pointer only counts for its own
 * workspace.
 */
export function studioProjectExpectedRevision(
  pointer: StudioProjectRevisionPointer | null,
  name: string,
): number | null {
  return pointer !== null && pointer.name === name ? pointer.revision : null;
}

/** Read the revision out of a load response, or null when it carries none. */
export function studioProjectRevisionFromLoadResponse(
  response: unknown,
  name: string,
): StudioProjectRevisionPointer | null {
  const revision = recordValue(response).revision;
  if (typeof revision !== "number" || !Number.isInteger(revision) || revision < 1) {
    return null;
  }
  return { name, revision };
}

/**
 * Report a refused save without adopting the revision that refused it.
 *
 * Taking the server's current revision here would make the next save
 * overwrite the other editor's work with the state this one still holds —
 * the lost update the conflict exists to prevent. The editor keeps its own
 * revision and the message says the workspace was left alone.
 *
 * A busy workspace (503) is a different answer and deserves a different one
 * back: another writer held the workspace for the whole of the server's wait,
 * nothing was written, and the very same save can be sent again. Reporting it
 * as an ordinary failure would push a user into reloading and reapplying work
 * that was never in conflict with anything.
 */
export function studioProjectSaveFailureState(
  error: unknown,
): StudioProjectFailureStatePatch {
  if (error instanceof StudioRequestError && error.status === 409) {
    return { error: `${error.message} Nothing was overwritten.` };
  }
  if (error instanceof StudioRequestError && error.status === 503) {
    return { error: `${error.message} Nothing was written; save again.` };
  }
  return studioProjectFailureState(error, "Project save failed");
}

export function studioProjectListLoadedState(
  serverProjects: ProjectSummary[],
): StudioProjectListLoadedStatePatch {
  return { serverProjects };
}

export function studioProjectDeletedListedState(
  deletedProjects: DeletedProjectSummary[],
): StudioProjectDeletedListedStatePatch {
  return { deletedProjects };
}

export function studioProjectRestoreState(
  projectState: StudioProjectStateSnapshot,
): StudioProjectStateSnapshot {
  return studioProjectSaveState(projectState);
}

export function studioProjectFailureState(
  error: unknown,
  fallbackMessage: string,
): StudioProjectFailureStatePatch {
  return {
    error: error instanceof Error && error.message.length > 0
      ? error.message
      : fallbackMessage,
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
