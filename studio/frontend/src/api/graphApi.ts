// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio frontend API
// Studio API: graph endpoints.
import { post, get } from "./http";
import type {
  PopulationNode,
  ProjectionEdge,
  NetworkGraph,
  GraphSimResult,
  NIRFormat,
} from "./types";

export const fetchGraphModels = () => get<string[]>("/graph/models");

export const createPopulation = (data: Partial<PopulationNode>) =>
  post<PopulationNode>("/graph/population", data);

export const createProjection = (data: { source_id: string; target_id: string; weight?: number; delay?: number; probability?: number }) =>
  post<ProjectionEdge>("/graph/projection", data);

export const validateGraph = (graph: NetworkGraph) =>
  post<{ valid: boolean; errors: string[] }>("/graph/validate", graph);

export const simulateGraph = (graph: NetworkGraph) =>
  post<GraphSimResult>("/graph/simulate", graph);

export const exportNIR = (graph: NetworkGraph) =>
  post<NIRFormat>("/graph/export-nir", graph);

export const importNIR = (nir: NIRFormat) =>
  post<NetworkGraph>("/graph/import-nir", nir);

