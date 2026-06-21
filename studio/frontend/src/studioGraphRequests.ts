// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio graph request builders

import type { NetworkGraph, PopulationNode, ProjectionEdge } from "./api/client";

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
