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
  PopulationCreateRequest,
  PopulationNode,
  ProjectionEdge,
  NetworkGraph,
  GraphSimResult,
  GraphValidation,
  PopulationModelContract,
  NIRExportResult,
  NIRImportRequest,
  NIRImportResult,
  ProjectionRule,
} from "./types";

/**
 * List the models a population may be built from.
 *
 * @returns Model names the graph routes accept.
 */
export const fetchGraphModels = () => get<string[]>("/graph/models");

/**
 * Create a population, letting the server assign its identity.
 *
 * Identity stays with the server so a population's id means the same thing in
 * every client that later reads the graph.
 *
 * @param data - The population to create.
 * @returns The population as the server created it, id included.
 */
export const createPopulation = (data: PopulationCreateRequest) =>
  post<PopulationNode>("/graph/population", data);

/**
 * Connect two populations, letting the server assign the projection's identity.
 *
 * The route does not carry `seed` or `autapses`; a caller that needs them
 * applies them to the returned edge, which is why they are absent here rather
 * than silently ignored.
 *
 * @param data - The endpoints and whatever connection parameters are set.
 * @returns The projection as the server created it, id included.
 */
export const createProjection = (data: {
  source_id: string;
  target_id: string;
  weight?: number;
  delay?: number;
  probability?: number;
  rule?: ProjectionRule;
}) => post<ProjectionEdge>("/graph/projection", data);

/**
 * Read what one population model will accept.
 *
 * The editor states each field's contract from this rather than from a copy in
 * the browser, so a model whose ranges change is edited against the new ones.
 *
 * @param name - The model's name.
 * @returns Its parameters, their ranges and their units.
 */
export const graphModelContract = (name: string) =>
  get<PopulationModelContract>(`/graph/models/${encodeURIComponent(name)}`);

/**
 * Ask the server what it refuses about a graph, without running it.
 *
 * Each issue names the field it is about, so the canvas can put the message on
 * the population or projection that caused it.
 *
 * @param graph - The graph to check.
 * @returns The issues, or that there are none.
 */
export const validateGraph = (graph: NetworkGraph) =>
  post<GraphValidation>("/graph/validate", graph);

/**
 * Run a graph.
 *
 * @param graph - The graph to run.
 * @returns Per-population spike counts and rates, or the refusal.
 */
export const simulateGraph = (graph: NetworkGraph) =>
  post<GraphSimResult>("/graph/simulate", graph);

/**
 * Export a graph as a real NIR file, the interchange format other tools read.
 *
 * @param graph - The graph to export, with its timestep, duration and seed.
 * @returns The file's bytes as base64, and what the file does not carry.
 */
export const exportNIR = (graph: NetworkGraph) =>
  post<NIRExportResult>("/graph/export-nir", graph);

/**
 * Import a NIR file, or a saved graph envelope, as a graph.
 *
 * @param request - The file's bytes as base64, or the envelope.
 * @returns The graph it describes, where it came from, and what the reading
 *   assumed.
 */
export const importNIR = (request: NIRImportRequest) =>
  post<NIRImportResult>("/graph/import-nir", request);
