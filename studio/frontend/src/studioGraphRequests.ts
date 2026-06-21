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
  PopulationNode,
  ProjectionEdge,
} from "./api/client";

export type StudioNeuronType = "excitatory" | "inhibitory";

export interface StudioPopulationCreateRequest extends Record<string, unknown> {
  label: string;
  model: string;
  count: number;
  neuron_type: StudioNeuronType;
  x: number;
  y: number;
}

export interface StudioProjectionCreateRequest {
  source_id: string;
  target_id: string;
  weight: number;
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
): NetworkGraph {
  return {
    populations,
    projections,
    duration,
    dt,
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
    model: "LIFNeuron",
    count: neuronType === "excitatory" ? 80 : 20,
    neuron_type: neuronType,
    x: 100 + index * 200,
    y: neuronType === "excitatory" ? 100 : 300,
  };
}

export function studioDefaultProjectionRequest(
  sourceId: string,
  targetId: string,
): StudioProjectionCreateRequest {
  return {
    source_id: sourceId,
    target_id: targetId,
    weight: 0.1,
    probability: 0.2,
  };
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
