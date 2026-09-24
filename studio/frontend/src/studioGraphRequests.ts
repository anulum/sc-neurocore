// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio graph request builders

/**
 * The requests the network canvas sends, and the patches its answers produce.
 *
 * Two conventions run through this file. **Layout is not identity**: where a
 * population sits on the canvas is a drawing concern, and a move writes a
 * position and nothing else, so dragging a node can never change what the
 * network computes. And **the sign of a weight follows the source population's
 * declared type** -- Dale's principle as the population states it -- with the
 * server refusing a sign that disagrees rather than flipping it, because a
 * silently corrected weight is a network the reader did not ask for.
 *
 * The defaults here are starting points for a reader who has just added a
 * node, not recommendations: 80 excitatory to 20 inhibitory is the ratio the
 * literature starts from, and every value is visible and editable in the
 * panel.
 */

import type {
  GraphSimResult,
  NetworkGraph,
  PipelineResult,
  PopulationDrive,
  PopulationNode,
  ProjectionEdge,
  ProjectionRule,
  StudioNeuronType,
} from "./api/client";

export type { StudioNeuronType } from "./api/client";

/** Catalogue model every new population starts from (exact-flow hard-reset LIF). */
/** The neuron model a new population starts with. */
export const STUDIO_DEFAULT_POPULATION_MODEL = "SCLapicqueLIFNeuron";
/**
 * Constant drive of a new excitatory population. With the default model
 * (v_rest 0, threshold 1, resistance 1, tau 20 ms) a current of 1.2 settles at
 * v_inf = 1.2 above threshold and fires regularly; inhibitory populations start
 * without external input and are driven through projections only.
 */
export const STUDIO_DEFAULT_EXCITATORY_DRIVE: PopulationDrive = { kind: "constant", current: 1.2 };
/**
 * Starting magnitude of a new projection weight. The public Network injects the
 * weight as drive for one timestep per source spike, so with dt 0.1 ms and
 * tau 20 ms one spike moves the default model by about weight × 0.005; 40 moves
 * it by a fifth of the threshold. This is a starting value, not a tuned one.
 */
export const STUDIO_DEFAULT_PROJECTION_WEIGHT = 40;
/** The connection probability a new random projection starts at. */
export const STUDIO_DEFAULT_PROJECTION_PROBABILITY = 0.2;

/** What the canvas sends to add a population. */
export interface StudioPopulationCreateRequest extends Record<string, unknown> {
  label: string;
  model: string;
  count: number;
  neuron_type: StudioNeuronType;
  x: number;
  y: number;
  drive: PopulationDrive;
}

/** What the canvas sends to add a projection. */
export interface StudioProjectionCreateRequest {
  source_id: string;
  target_id: string;
  weight: number;
  delay: number;
  rule: ProjectionRule;
  probability: number;
}

/** A graph as the canvas holds it: its populations and its projections. */
export interface StudioGraphElements {
  populations: PopulationNode[];
  projections: ProjectionEdge[];
}

/** A graph request has started: clear the error, show it running. */
export interface StudioGraphBusyStatePatch {
  error: null;
  graphErrors?: [];
  /** Cleared with the messages: a located failure outlives nothing else. */
  graphIssues?: [];
  graphSimResult?: null;
  isSimulating: true;
  pipelineResult?: null;
}

/** The pipeline finished, with its result. */
export interface StudioPipelineCompletedStatePatch {
  isSimulating: false;
  pipelineResult: PipelineResult;
}

/** The catalogue of models a population may use arrived. */
export interface StudioGraphModelsLoadedStatePatch {
  graphModels: string[];
}

/** A population was added. */
export interface StudioPopulationAddedStatePatch {
  graphPopulations: PopulationNode[];
}

/** A population changed. */
export interface StudioPopulationUpdatedStatePatch {
  graphPopulations: PopulationNode[];
}

/** A projection was added. */
export interface StudioProjectionAddedStatePatch {
  graphProjections: ProjectionEdge[];
}

/** A projection changed. */
export interface StudioProjectionUpdatedStatePatch {
  graphProjections: ProjectionEdge[];
}

/** A projection was removed. */
export interface StudioProjectionRemovedStatePatch {
  graphProjections: ProjectionEdge[];
}

/** A graph simulation finished, with its result. */
export interface StudioGraphSimulationCompletedStatePatch {
  graphSimResult: GraphSimResult;
  isSimulating: false;
}

/**
 * A whole graph was imported, replacing what was on the canvas.
 *
 * The timestep, duration and seed come with it when the graph states them: a
 * projection delay is a whole number of the graph's timesteps, and unstated
 * seeds are derived from the graph seed, so the network is only the same
 * network under them.
 */
export interface StudioGraphImportedStatePatch {
  activeTab: "canvas";
  graphPopulations: PopulationNode[];
  graphProjections: ProjectionEdge[];
  dt?: number;
  duration?: number;
  seed?: number;
}

/** A graph request failed, with the message to show. */
export interface StudioGraphFailureStatePatch {
  error: string;
  isSimulating?: false;
}

/**
 * Build the graph payload a run is submitted with.
 *
 * The seed is omitted rather than sent as null when there is none, so the
 * server draws its own and the request says what it means.
 *
 * @param populations - The populations.
 * @param projections - The projections between them.
 * @param duration - How long to run, in milliseconds.
 * @param dt - The integration step, in milliseconds.
 * @param seed - The seed, or `null` to let the server choose.
 * @returns The graph to submit.
 */
export function studioGraphRequest(
  populations: PopulationNode[],
  projections: ProjectionEdge[],
  duration: number,
  dt: number,
  seed: number | null = null,
): NetworkGraph {
  return {
    populations,
    projections,
    duration,
    dt,
    ...(seed === null ? {} : { seed }),
  };
}

/**
 * The pipeline has started.
 *
 * @returns The patch.
 */
export function studioPipelineStartState(): StudioGraphBusyStatePatch {
  return {
    error: null,
    isSimulating: true,
    pipelineResult: null,
  };
}

/**
 * The pipeline finished.
 *
 * @param pipelineResult - What it produced.
 * @returns The patch.
 */
export function studioPipelineCompletedState(
  pipelineResult: PipelineResult,
): StudioPipelineCompletedStatePatch {
  return {
    isSimulating: false,
    pipelineResult,
  };
}

/**
 * A graph simulation has started.
 *
 * @returns The patch.
 */
export function studioGraphSimulationStartState(): StudioGraphBusyStatePatch {
  return {
    error: null,
    graphErrors: [],
    graphIssues: [],
    isSimulating: true,
  };
}

/**
 * A graph simulation finished.
 *
 * @param graphSimResult - What it produced.
 * @returns The patch.
 */
export function studioGraphSimulationCompletedState(
  graphSimResult: GraphSimResult,
): StudioGraphSimulationCompletedStatePatch {
  return {
    graphSimResult,
    isSimulating: false,
  };
}

/**
 * The model catalogue arrived.
 *
 * @param graphModels - The models a population may use.
 * @returns The patch.
 */
export function studioGraphModelsLoadedState(
  graphModels: string[],
): StudioGraphModelsLoadedStatePatch {
  return { graphModels };
}

/**
 * A population was added.
 *
 * @param graphPopulations - The populations as they stand.
 * @param population - The population the server created.
 * @returns The patch.
 */
export function studioPopulationAddedState(
  graphPopulations: PopulationNode[],
  population: PopulationNode,
): StudioPopulationAddedStatePatch {
  return {
    graphPopulations: [...graphPopulations, population],
  };
}

/**
 * A population changed.
 *
 * @param graphPopulations - The populations as they stand.
 * @param populationId - The population that changed.
 * @param updates - What changed about it.
 * @returns The patch.
 */
export function studioPopulationUpdatedState(
  graphPopulations: PopulationNode[],
  populationId: string,
  updates: Partial<PopulationNode>,
): StudioPopulationUpdatedStatePatch {
  return {
    graphPopulations: graphPopulations.map((population) =>
      population.id === populationId ? { ...population, ...updates } : population),
  };
}

/**
 * A projection was added.
 *
 * @param graphProjections - The projections as they stand.
 * @param projection - The projection the server created.
 * @returns The patch.
 */
export function studioProjectionAddedState(
  graphProjections: ProjectionEdge[],
  projection: ProjectionEdge,
): StudioProjectionAddedStatePatch {
  return {
    graphProjections: [...graphProjections, projection],
  };
}

/**
 * A projection changed.
 *
 * @param graphProjections - The projections as they stand.
 * @param projectionId - The projection that changed.
 * @param updates - What changed about it.
 * @returns The patch.
 */
export function studioProjectionUpdatedState(
  graphProjections: ProjectionEdge[],
  projectionId: string,
  updates: Partial<ProjectionEdge>,
): StudioProjectionUpdatedStatePatch {
  return {
    graphProjections: graphProjections.map((projection) =>
      projection.id === projectionId ? { ...projection, ...updates } : projection),
  };
}

/**
 * A projection was removed.
 *
 * @param graphProjections - The projections as they stand.
 * @param projectionId - The projection that is gone.
 * @returns The patch.
 */
export function studioProjectionRemovedState(
  graphProjections: ProjectionEdge[],
  projectionId: string,
): StudioProjectionRemovedStatePatch {
  return {
    graphProjections: graphProjections.filter((projection) => projection.id !== projectionId),
  };
}

/**
 * A graph was imported.
 *
 * @param graph - The imported network.
 * @returns The patch, replacing the canvas entirely.
 */
export function studioGraphImportedState(graph: NetworkGraph): StudioGraphImportedStatePatch {
  return {
    activeTab: "canvas",
    graphPopulations: graph.populations,
    graphProjections: graph.projections,
    ...(graph.dt === undefined ? {} : { dt: graph.dt }),
    ...(graph.duration === undefined ? {} : { duration: graph.duration }),
    ...(graph.seed === undefined ? {} : { seed: graph.seed }),
  };
}

/**
 * A graph request failed.
 *
 * @param error - What was thrown, which need not be an `Error`.
 * @param fallbackMessage - What to say when it carries no message.
 * @param options - Whether this failure also ends a run that was showing
 *   as busy.
 * @returns The patch.
 */
export function studioGraphFailureState(
  error: unknown,
  fallbackMessage: string,
  options: { clearBusy?: boolean } = {},
): StudioGraphFailureStatePatch {
  return {
    error: error instanceof Error && error.message.length > 0
      ? error.message
      : fallbackMessage,
    ...(options.clearBusy ? { isSimulating: false } : {}),
  };
}

/**
 * Build the request for a population the reader just added.
 *
 * The counts and the position are starting points, all of them visible and
 * editable in the panel: 80 excitatory against 20 inhibitory is where the
 * literature starts, not a recommendation this build is making.
 *
 * @param neuronType - Whether it excites or inhibits.
 * @param index - Which population this is, which sets where it is drawn.
 * @returns The request.
 */
export function studioDefaultPopulationRequest(
  neuronType: StudioNeuronType,
  index: number,
): StudioPopulationCreateRequest {
  return {
    label: neuronType === "excitatory" ? `Exc ${index}` : `Inh ${index}`,
    model: STUDIO_DEFAULT_POPULATION_MODEL,
    count: neuronType === "excitatory" ? 80 : 20,
    neuron_type: neuronType,
    x: 100 + index * 200,
    y: neuronType === "excitatory" ? 100 : 300,
    drive: neuronType === "excitatory" ? STUDIO_DEFAULT_EXCITATORY_DRIVE : { kind: "none" },
  };
}

/**
 * Build the request for a projection the reader just drew.
 *
 * The weight's sign follows the source population's declared type -- Dale's
 * principle as the population states it. The server refuses a sign that
 * disagrees rather than flipping it, because a silently corrected weight is a
 * network the reader did not ask for.
 *
 * @param sourceId - The population the projection leaves.
 * @param targetId - The population it reaches.
 * @param sourceNeuronType - What the source population declares itself to be.
 * @returns The request.
 */
export function studioDefaultProjectionRequest(
  sourceId: string,
  targetId: string,
  sourceNeuronType: StudioNeuronType,
): StudioProjectionCreateRequest {
  return {
    source_id: sourceId,
    target_id: targetId,
    weight: sourceNeuronType === "inhibitory"
      ? -STUDIO_DEFAULT_PROJECTION_WEIGHT
      : STUDIO_DEFAULT_PROJECTION_WEIGHT,
    delay: 0,
    rule: "random",
    probability: STUDIO_DEFAULT_PROJECTION_PROBABILITY,
  };
}

/**
 * Describe a population's input in one phrase.
 *
 * @param drive - The drive, if it has one.
 * @returns The phrase the canvas shows.
 */
export function studioPopulationDriveLabel(drive: PopulationDrive | undefined): string {
  if (!drive || drive.kind === "none") return "no input";
  if (drive.kind === "constant") return `I = ${drive.current}`;
  return `Poisson ${drive.rate_hz} Hz × ${drive.weight}`;
}

/**
 * Describe a projection in one phrase: its weight, its rule, its delay.
 *
 * A zero delay is left out rather than shown as `d=0ms`, because every
 * projection has a delay and only a non-zero one tells the reader
 * something.
 *
 * @param projection - The projection.
 * @returns The phrase the canvas shows.
 */
export function studioProjectionLabel(projection: ProjectionEdge): string {
  const rule = projection.rule === "all_to_all" ? "all" : `p=${projection.probability}`;
  const delay = projection.delay > 0 ? ` d=${projection.delay}ms` : "";
  return `w=${projection.weight} ${rule}${delay}`;
}

/** One population's new layout position after a canvas interaction. */
export interface StudioNodeMove {
  id: string;
  position: { x: number; y: number };
}

/**
 * What a batch of canvas node changes means for the graph.
 *
 * The canvas used to compute the new node array and then write back positions
 * only, so a delete was computed and discarded: the population and every
 * projection touching it stayed in the graph and reappeared on the next render.
 * Separating the two answers keeps layout out of scientific identity — a move
 * changes where a node is drawn and nothing else — while making a removal an
 * explicit instruction the store can act on.
 */
export interface StudioNodeChangePlan {
  moved: StudioNodeMove[];
  removed: string[];
}

/**
 * Read a batch of canvas node changes as moves and removals.
 *
 * A position is reported only when it actually differs, so a render that
 * reasserts the same layout writes nothing.
 *
 * @param changes - The change batch the canvas produced.
 * @param nextNodes - The nodes as the canvas library now holds them.
 * @param populations - The graph as it stands.
 * @returns Which populations moved where, and which were removed.
 */
export function studioNodeChangePlan(
  changes: readonly { type: string; id?: string }[],
  nextNodes: readonly { id: string; position: { x: number; y: number } }[],
  populations: readonly PopulationNode[],
): StudioNodeChangePlan {
  const removed = changes
    .filter((change) => change.type === "remove")
    .map((change) => change.id)
    .filter((id): id is string => typeof id === "string" && id.length > 0);
  const removedSet = new Set(removed);
  const moved: StudioNodeMove[] = [];
  for (const node of nextNodes) {
    if (removedSet.has(node.id)) {
      continue;
    }
    const population = populations.find((candidate) => candidate.id === node.id);
    if (
      population &&
      (population.position.x !== node.position.x || population.position.y !== node.position.y)
    ) {
      moved.push({ id: node.id, position: node.position });
    }
  }
  return { moved, removed };
}

/**
 * Remove a population and everything attached to it.
 *
 * Both directions are removed. A projection whose source or target is gone
 * is not a projection, and leaving it would put an edge on the canvas with
 * one end attached to nothing.
 *
 * @param graph - The graph as it stands.
 * @param populationId - The population to remove.
 * @returns The graph without it.
 */
export function studioGraphWithoutPopulation(
  graph: StudioGraphElements,
  populationId: string,
): StudioGraphElements {
  return {
    populations: graph.populations.filter((population) => population.id !== populationId),
    projections: graph.projections.filter((projection) =>
      projection.source !== populationId && projection.target !== populationId),
  };
}
