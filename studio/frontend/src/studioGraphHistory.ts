// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Undo history for Network Canvas graph edits

/**
 * Undo and redo for the graph the canvas edits.
 *
 * Deleting a population removes it and every projection touching it, which is
 * the correct behaviour and also the one a user most wants back. The way back
 * used to be a saved workspace revision, which loses everything done since.
 *
 * The history holds whole graph snapshots rather than inverse operations. A
 * snapshot is two arrays of plain objects, and the alternative — an inverse per
 * operation — has to be right for every operation separately, including the
 * ones that touch several objects at once. Reinstating a snapshot is right by
 * construction.
 *
 * A move is deliberately not an edit. Dragging a node writes a position on
 * every frame, and recording those would bury the edits a user actually wants
 * back under layout noise; the canvas keeps drag and view changes out of
 * scientific identity, and so does this.
 */

import type { PopulationNode, ProjectionEdge } from "./api/client";

/**
 * How many edits the history keeps.
 *
 * Deep enough that a working session is recoverable, bounded so a long session
 * cannot grow the store without limit.
 */
export const STUDIO_GRAPH_HISTORY_LIMIT = 50;

/** The whole editable graph at one moment. */
export interface StudioGraphSnapshot {
  populations: PopulationNode[];
  projections: ProjectionEdge[];
}

/** Snapshots before and after the current graph. */
export interface StudioGraphHistory {
  past: StudioGraphSnapshot[];
  future: StudioGraphSnapshot[];
}

/**
 * The history a session starts with.
 *
 * @returns A history with nothing to step in either direction.
 */
export function emptyGraphHistory(): StudioGraphHistory {
  return { future: [], past: [] };
}

/**
 * Return the current graph as a snapshot.
 *
 * The arrays are copied so a later mutation of the live state cannot reach
 * back into a recorded snapshot; the objects inside are treated as immutable,
 * which is how the store's reducers already produce them.
 *
 * @param populations - Populations as the graph holds them.
 * @param projections - Projections as the graph holds them.
 * @returns A snapshot holding copies of both arrays, so a later mutation of the live state cannot reach back into it.
 */
export function graphSnapshotOf(
  populations: readonly PopulationNode[],
  projections: readonly ProjectionEdge[],
): StudioGraphSnapshot {
  return { populations: [...populations], projections: [...projections] };
}

/**
 * Return whether two snapshots describe the same graph.
 *
 * Used so an edit that changed nothing records nothing: an undo should reach
 * the last real change, not an empty step.
 *
 * @param left - One snapshot.
 * @param right - The other snapshot.
 * @returns Whether the two snapshots hold the same graph.
 */
export function graphSnapshotsEqual(
  left: StudioGraphSnapshot,
  right: StudioGraphSnapshot,
): boolean {
  return (
    left.populations.length === right.populations.length &&
    left.projections.length === right.projections.length &&
    left.populations.every((population, index) => population === right.populations[index]) &&
    left.projections.every((projection, index) => projection === right.projections[index])
  );
}

/**
 * Record the graph as it stood before an edit.
 *
 * The redo branch is discarded, because editing after an undo replaces the
 * future that undo had stepped back from; offering a redo into a graph that no
 * longer follows from the current one would reinstate work the user has
 * already moved past. The oldest entry is dropped once the limit is reached.
 *
 * @param history - The history as it stands.
 * @param before - The graph as it stood before the edit.
 * @returns The history with the before-state pushed and the redo branch dropped.
 */
export function graphEditRecorded(
  history: StudioGraphHistory,
  before: StudioGraphSnapshot,
): StudioGraphHistory {
  const past = [...history.past, before];
  return {
    future: [],
    past: past.length > STUDIO_GRAPH_HISTORY_LIMIT ? past.slice(past.length - STUDIO_GRAPH_HISTORY_LIMIT) : past,
  };
}

/** One step of the history, and the graph it reinstates. */
export interface StudioGraphHistoryStep {
  history: StudioGraphHistory;
  snapshot: StudioGraphSnapshot;
}

/**
 * Step back one edit, or return `null` when there is nothing to undo.
 *
 * @param history - The history as it stands.
 * @param current - The graph being stepped away from, kept for redo.
 * @returns The step to apply and the history after it, or `null` when there is nothing to undo.
 */
export function graphUndone(
  history: StudioGraphHistory,
  current: StudioGraphSnapshot,
): StudioGraphHistoryStep | null {
  const snapshot = history.past[history.past.length - 1];
  // An empty history yields `undefined` here.
  if (snapshot === undefined) {
    return null;
  }
  return {
    history: {
      future: [current, ...history.future],
      past: history.past.slice(0, -1),
    },
    snapshot,
  };
}

/**
 * Step forward one edit, or return `null` when there is nothing to redo.
 *
 * @param history - The history as it stands.
 * @param current - The graph being stepped away from, kept for undo.
 * @returns The step to apply and the history after it, or `null` when there is nothing to redo.
 */
export function graphRedone(
  history: StudioGraphHistory,
  current: StudioGraphSnapshot,
): StudioGraphHistoryStep | null {
  const [snapshot, ...future] = history.future;
  // An empty history yields `undefined` here.
  if (snapshot === undefined) {
    return null;
  }
  return {
    history: { future, past: [...history.past, current] },
    snapshot,
  };
}

/**
 * Return whether an update to a population is layout only.
 *
 * A drag writes a position on every frame. Those are not edits to undo.
 *
 * @param updates - The fields one canvas update would write.
 * @returns Whether the update changes only where a node is drawn.
 */
export function isLayoutOnlyUpdate(updates: Partial<PopulationNode>): boolean {
  const keys = Object.keys(updates);
  return keys.length > 0 && keys.every((key) => key === "position");
}
