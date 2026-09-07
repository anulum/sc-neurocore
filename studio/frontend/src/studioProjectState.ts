// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio project state snapshot helpers

/**
 * Turning the store into a saved workspace, and a saved workspace back.
 *
 * The two directions are deliberately asymmetric. Saving takes a typed
 * snapshot of what the store holds. Loading reads a document that may have
 * been written by an older build, so every field goes through an accessor with
 * a fallback and anything unrecognised is dropped rather than trusted: a
 * workspace saved a year ago must open, not throw, and must not carry a value
 * this build cannot use.
 */

import { StudioRequestError } from "./api/client";
import type {
  DeletedProjectSummary,
  PopulationNode,
  ProjectSaveResponse,
  ProjectSummary,
  ProjectionEdge,
} from "./api/client";
import type { StudioSimulationSourceMode } from "./studioSimulationConfig";

/** The training settings a workspace carries. */
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

/** Everything the store contributes to a snapshot. */
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

/** A workspace as it is stored. */
export interface StudioProjectStateSnapshot
  extends StudioProjectSnapshotInput, Record<string, unknown> {}

/** Which workspace revision the editor is currently working from. */
export interface StudioProjectRevisionPointer {
  name: string;
  revision: number;
}

/** A save succeeded: the revision it wrote and when. */
export interface StudioProjectSavedStatePatch {
  projectSaveResult: ProjectSaveResponse;
  projectRevision: StudioProjectRevisionPointer;
}

/** A deleted workspace came back, under the revision it returned at. */
export interface StudioProjectRestorePointerPatch {
  projectRevision: StudioProjectRevisionPointer | null;
}

/** The stored workspaces arrived. */
export interface StudioProjectListLoadedStatePatch {
  serverProjects: ProjectSummary[];
}

/** The recoverable trash arrived. */
export interface StudioProjectDeletedListedStatePatch {
  deletedProjects: DeletedProjectSummary[];
}

/** A workspace operation failed, with the message to show. */
export interface StudioProjectFailureStatePatch {
  error: string;
  /**
   * The workspace and revision a refused edit diverged from, when the refusal
   * was a save conflict. Present only then: it is what
   * `keepRefusedEdit` needs, and its absence is how the UI knows there is
   * nothing to keep.
   */
  refusedEdit?: { name: string; baseRevision: number } | null;
}

/**
 * Take a snapshot of the store, for saving.
 *
 * @param input - What the store holds.
 * @returns The workspace to store.
 */
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

/**
 * Record a successful save.
 *
 * @param projectSaveResult - What the server wrote.
 * @returns The patch.
 */
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
 *
 * @param pointer - The revision the editor loaded, if it loaded one.
 * @param name - The workspace being saved under.
 * @returns The revision to claim, or `null` for a new workspace.
 */
export function studioProjectExpectedRevision(
  pointer: StudioProjectRevisionPointer | null,
  name: string,
): number | null {
  return pointer !== null && pointer.name === name ? pointer.revision : null;
}

/**
 * Read the revision out of a load response, or `null` when it carries none.
 *
 * @param response - The load response.
 * @param name - The workspace it was loaded for.
 * @returns The pointer, or `null`.
 */
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
 *
 * @param error - Whatever the save threw or rejected with.
 * @param name - The workspace the save was for.
 * @param baseRevision - The revision the edit started from, or `null`.
 * @returns The patch, with the message the reader should see.
 */
export function studioProjectSaveFailureState(
  error: unknown,
  name = "",
  baseRevision: number | null = null,
): StudioProjectFailureStatePatch {
  if (error instanceof StudioRequestError && error.status === 409) {
    // The refusal protected the other editor. It did not keep this one's work,
    // which still exists only here — so say that it can be kept, and carry what
    // keeping it needs.
    return {
      error: `${error.message} Nothing was overwritten; this edit can be kept as its own branch.`,
      refusedEdit: baseRevision === null ? null : { name, baseRevision },
    };
  }
  if (error instanceof StudioRequestError && error.status === 503) {
    return { error: `${error.message} Nothing was written; save again.` };
  }
  return studioProjectFailureState(error, "Project save failed");
}

/**
 * Take the stored workspaces into the store.
 *
 * @param serverProjects - The workspaces as the server listed them.
 * @returns The patch.
 */
export function studioProjectListLoadedState(
  serverProjects: ProjectSummary[],
): StudioProjectListLoadedStatePatch {
  return { serverProjects };
}

/**
 * Take the recoverable trash into the store.
 *
 * @param deletedProjects - The deleted workspaces.
 * @returns The patch.
 */
export function studioProjectDeletedListedState(
  deletedProjects: DeletedProjectSummary[],
): StudioProjectDeletedListedStatePatch {
  return { deletedProjects };
}

/**
 * Record that a deleted workspace came back.
 *
 * @param projectState - The restored workspace's name and revision.
 * @returns The patch.
 */
export function studioProjectRestoreState(
  projectState: StudioProjectStateSnapshot,
): StudioProjectStateSnapshot {
  return studioProjectSaveState(projectState);
}

/**
 * Report a failed workspace operation.
 *
 * @param error - Whatever was thrown or rejected.
 * @param fallbackMessage - What to show when the error carries no message.
 * @returns The patch.
 */
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

/**
 * Read a stored workspace into the store, tolerating an older shape.
 *
 * Every field is read through an accessor with a fallback, so a workspace
 * written by an earlier build opens rather than throwing, and a value this
 * build cannot use is dropped rather than carried into a request.
 *
 * @param response - The stored document.
 * @param fallbackTrainingConfig - The training settings to fall back to, field
 *   by field, for anything the document does not carry.
 * @returns The patch.
 */
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

/**
 * Read a stored object, or an empty one.
 *
 * @param value - The stored value.
 * @returns The object.
 */
function recordValue(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

/**
 * Read the stored source mode, defaulting to a catalogue model.
 *
 * @param value - The stored value.
 * @returns The mode.
 */
function sourceModeValue(value: unknown): StudioSimulationSourceMode {
  return value === "ode" ? "ode" : "model";
}

/**
 * Read a stored string, falling back when it is not one.
 *
 * @param value - The stored value.
 * @param fallback - What to use instead.
 * @returns The string.
 */
function stringValue(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}

/**
 * Read a stored number, falling back for anything not finite.
 *
 * @param value - The stored value.
 * @param fallback - What to use instead.
 * @returns The number.
 */
function finiteNumberValue(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

/**
 * Read a stored number that has to be above zero.
 *
 * A stored zero would divide or step by nothing; the fallback is used instead.
 *
 * @param value - The stored value.
 * @param fallback - What to use instead.
 * @returns The number.
 */
function positiveNumberValue(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : fallback;
}

/**
 * Read a stored boolean, falling back when it is not one.
 *
 * @param value - The stored value.
 * @param fallback - What to use instead.
 * @returns The boolean.
 */
function booleanValue(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

/**
 * Read a stored list of strings, dropping anything that is not one.
 *
 * @param value - The stored value.
 * @returns The strings.
 */
function stringArrayValue(value: unknown): string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string") ? value : [];
}

/**
 * Read a stored record of numbers, dropping entries that are not.
 *
 * @param value - The stored value.
 * @returns The record.
 */
function numberRecordValue(value: unknown): Record<string, number> {
  const record = recordValue(value);
  return Object.fromEntries(
    Object.entries(record).filter((entry): entry is [string, number] =>
      typeof entry[1] === "number" && Number.isFinite(entry[1])),
  );
}

/**
 * Read a stored list of numbers, falling back when it is not one.
 *
 * @param value - The stored value.
 * @param fallback - What to use instead.
 * @returns The numbers.
 */
function numberArrayValue(value: unknown, fallback: number[]): number[] {
  if (!Array.isArray(value)) return fallback;
  const numbers: number[] = [];
  for (const item of value as unknown[]) {
    // A single non-finite entry disqualifies the list: a trace with one `NaN`
    // in it is not a trace that can be plotted or sent.
    if (typeof item !== "number" || !Number.isFinite(item)) return fallback;
    numbers.push(item);
  }
  return numbers;
}

/**
 * Read the stored populations, dropping any that are not shaped like one.
 *
 * @param value - The stored value.
 * @returns The populations.
 */
function populationArrayValue(value: unknown): PopulationNode[] {
  return Array.isArray(value) ? value.filter(isRecord) as unknown as PopulationNode[] : [];
}

/**
 * Read the stored projections, dropping any that are not shaped like one.
 *
 * @param value - The stored value.
 * @returns The projections.
 */
function projectionArrayValue(value: unknown): ProjectionEdge[] {
  return Array.isArray(value) ? value.filter(isRecord) as unknown as ProjectionEdge[] : [];
}

/**
 * Whether a value is a plain object, as against an array or `null`.
 *
 * @param value - The value.
 * @returns Whether it is a record.
 */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Read the stored training settings, field by field.
 *
 * @param value - The stored value.
 * @param fallback - The settings to fall back to, field by field.
 * @returns The settings.
 */
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
