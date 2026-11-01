// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Saved graph workspace and fresh restoration context

import type { PopulationNode, ProjectionEdge } from "./api/types";
import { graphWorkspaceId, readGraphJobRecovery, type GraphJobRecovery } from "./graphJobRecovery";
import { readGraphJobSavedInput } from "./graphJobSavedInput";
import { emptyGraphHistory, type StudioGraphHistory } from "./studioGraphHistory";

/** Graph fields carried by current workspace snapshots. */
export interface StudioGraphWorkspaceSavedState {
  graphPopulations: PopulationNode[];
  graphProjections: ProjectionEdge[];
  graphWorkspaceId: string;
  graphJobRecovery: GraphJobRecovery | null;
}

/** Legacy snapshot callers may have no graph fields yet. */
export type StudioGraphWorkspaceInput = Partial<StudioGraphWorkspaceSavedState>;

/** Local restore generation; explicit snapshots take precedence over the tab's latest cache. */
export interface StudioGraphWorkspaceContext {
  restored: boolean;
}

/** A restored graph has a new local context and no trusted result or previous editor history. */
export interface StudioGraphWorkspaceRestoreState extends StudioGraphWorkspaceSavedState {
  graphContextToken: StudioGraphWorkspaceContext;
  graphSimResult: null;
  graphHistory: StudioGraphHistory;
  selectedPopulationId: null;
  selectedPopulationIds: string[];
  selectedProjectionId: null;
}

/**
 * Retain graph inputs and recovery metadata without saving runtime result data.
 *
 * @param input - Current store, or an older snapshot caller without a graph.
 * @returns Detached graph fields when present, with their stable workspace identity.
 */
export function saveStudioGraphWorkspace(input: StudioGraphWorkspaceInput): StudioGraphWorkspaceInput {
  if (input.graphPopulations === undefined && input.graphProjections === undefined) return {};
  const graph = readGraphJobSavedInput({ populations: input.graphPopulations ?? [], projections: input.graphProjections ?? [] });
  return {
    graphPopulations: graph.populations, graphProjections: graph.projections,
    graphWorkspaceId: graphWorkspaceId(input.graphWorkspaceId),
    graphJobRecovery: readGraphJobRecovery(input.graphJobRecovery),
  };
}

/**
 * Reopen a saved graph without carrying another workspace's result or asynchronous context.
 *
 * @param state - Untrusted saved project or named browser session.
 * @returns Original graph/recovery values, a fresh context token and empty local editor history.
 */
export function restoreStudioGraphWorkspace(state: Record<string, unknown>): StudioGraphWorkspaceRestoreState {
  const graph = readGraphJobSavedInput({ populations: state.graphPopulations === undefined ? [] : state.graphPopulations,
    projections: state.graphProjections === undefined ? [] : state.graphProjections });
  return {
    graphPopulations: graph.populations, graphProjections: graph.projections,
    graphWorkspaceId: graphWorkspaceId(state.graphWorkspaceId),
    graphJobRecovery: readGraphJobRecovery(state.graphJobRecovery),
    graphContextToken: { restored: true }, graphSimResult: null, graphHistory: emptyGraphHistory(),
    selectedPopulationId: null, selectedPopulationIds: [], selectedProjectionId: null,
  };
}
