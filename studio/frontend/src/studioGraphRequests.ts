// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio graph request builders

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
export const STUDIO_DEFAULT_PROJECTION_PROBABILITY = 0.2;

export interface StudioPopulationCreateRequest extends Record<string, unknown> {
  label: string;
  model: string;
  count: number;
  neuron_type: StudioNeuronType;
  x: number;
  y: number;
  drive: PopulationDrive;
}

export interface StudioProjectionCreateRequest {
  source_id: string;
  target_id: string;
  weight: number;
  delay: number;
  rule: ProjectionRule;
  probability: number;
}

export interface StudioGraphElements {
  populations: PopulationNode[];
  projections: ProjectionEdge[];
}

export interface StudioGraphBusyStatePatch {
  error: null;
  graphErrors?: [];
  graphSimResult?: null;
  isSimulating: true;
  pipelineResult?: null;
}

export interface StudioPipelineCompletedStatePatch {
  isSimulating: false;
  pipelineResult: PipelineResult;
}

export interface StudioGraphModelsLoadedStatePatch {
  graphModels: string[];
}

export interface StudioPopulationAddedStatePatch {
  graphPopulations: PopulationNode[];
}

export interface StudioPopulationUpdatedStatePatch {
  graphPopulations: PopulationNode[];
}

export interface StudioProjectionAddedStatePatch {
  graphProjections: ProjectionEdge[];
}

export interface StudioProjectionUpdatedStatePatch {
  graphProjections: ProjectionEdge[];
}

export interface StudioProjectionRemovedStatePatch {
  graphProjections: ProjectionEdge[];
}

export interface StudioGraphValidationFailedStatePatch {
  graphErrors: string[];
  isSimulating: false;
}

export interface StudioGraphSimulationCompletedStatePatch {
  graphSimResult: GraphSimResult;
  isSimulating: false;
}

export interface StudioGraphImportedStatePatch {
  activeTab: "canvas";
  graphPopulations: PopulationNode[];
  graphProjections: ProjectionEdge[];
}

export interface StudioGraphFailureStatePatch {
  error: string;
  isSimulating?: false;
}

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

export function studioPipelineStartState(): StudioGraphBusyStatePatch {
  return {
    error: null,
    isSimulating: true,
    pipelineResult: null,
  };
}

export function studioPipelineCompletedState(
  pipelineResult: PipelineResult,
): StudioPipelineCompletedStatePatch {
  return {
    isSimulating: false,
    pipelineResult,
  };
}

export function studioGraphSimulationStartState(): StudioGraphBusyStatePatch {
  return {
    error: null,
    graphErrors: [],
    isSimulating: true,
  };
}

export function studioGraphValidationFailedState(
  graphErrors: string[],
): StudioGraphValidationFailedStatePatch {
  return {
    graphErrors,
    isSimulating: false,
  };
}

export function studioGraphSimulationCompletedState(
  graphSimResult: GraphSimResult,
): StudioGraphSimulationCompletedStatePatch {
  return {
    graphSimResult,
    isSimulating: false,
  };
}

export function studioGraphModelsLoadedState(
  graphModels: string[],
): StudioGraphModelsLoadedStatePatch {
  return { graphModels };
}

export function studioPopulationAddedState(
  graphPopulations: PopulationNode[],
  population: PopulationNode,
): StudioPopulationAddedStatePatch {
  return {
    graphPopulations: [...graphPopulations, population],
  };
}

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

export function studioProjectionAddedState(
  graphProjections: ProjectionEdge[],
  projection: ProjectionEdge,
): StudioProjectionAddedStatePatch {
  return {
    graphProjections: [...graphProjections, projection],
  };
}

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

export function studioProjectionRemovedState(
  graphProjections: ProjectionEdge[],
  projectionId: string,
): StudioProjectionRemovedStatePatch {
  return {
    graphProjections: graphProjections.filter((projection) => projection.id !== projectionId),
  };
}

export function studioGraphImportedState(nir: NetworkGraph): StudioGraphImportedStatePatch {
  return {
    activeTab: "canvas",
    graphPopulations: nir.populations,
    graphProjections: nir.projections,
  };
}

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
 * Default projection request. The weight sign follows the source population's
 * declared type (Dale's principle as the population states it); the server
 * rejects a sign that disagrees instead of flipping it.
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

export function studioPopulationDriveLabel(drive: PopulationDrive | undefined): string {
  if (!drive || drive.kind === "none") return "no input";
  if (drive.kind === "constant") return `I = ${drive.current}`;
  return `Poisson ${drive.rate_hz} Hz × ${drive.weight}`;
}

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
 * `nextNodes` is the array the canvas library produced from the change batch;
 * `populations` is the graph as it stands. A position is reported only when it
 * actually differs, so a render that reasserts the same layout writes nothing.
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
