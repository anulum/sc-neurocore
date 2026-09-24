// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Topology of the graph as a table a screen reader can read

/**
 * The network as text, for readers who cannot see the canvas.
 *
 * A node-and-edge diagram carries its meaning in positions and arrows. Neither
 * survives a screen reader: the canvas reports a list of draggable boxes and
 * says nothing about what is connected to what. The graph's topology is a
 * finite relation, so it can be stated exactly rather than approximated, and a
 * table states it — one row per population, with what reaches it and what it
 * reaches.
 *
 * Every string here is derived from the same graph the canvas draws and the
 * server runs. It is a second presentation of one truth, never a summary that
 * could drift from it.
 */

import type { PopulationNode, ProjectionEdge } from "./api/client";
import { studioPopulationDriveLabel, studioProjectionLabel } from "./studioGraphRequests";
import { studioGraphIssuesById, type StudioGraphIssueLocation } from "./studioGraphValidation";

/** One connection as a row states it. */
export interface StudioGraphTableConnection {
  /** Projection identifier, so a row can name what to remove. */
  id: string;
  /** The population at the other end. */
  populationLabel: string;
  /** Weight, rule and delay, in the canvas's own words. */
  detail: string;
  /** What validation said about this projection, if anything. */
  issues: string[];
}

/** One population and everything the graph says about it. */
export interface StudioGraphTableRow {
  id: string;
  label: string;
  model: string;
  count: number;
  neuronType: string;
  drive: string;
  incoming: StudioGraphTableConnection[];
  outgoing: StudioGraphTableConnection[];
  /** One sentence a screen reader can read without visiting the cells. */
  description: string;
  /** What validation said about this population, if anything. */
  issues: string[];
}

/** The whole table, with the caption that describes the topology. */
export interface StudioGraphTable {
  caption: string;
  rows: StudioGraphTableRow[];
}

/** Column headers, in the order the rows carry them. */
export const STUDIO_GRAPH_TABLE_COLUMNS = [
  "Population",
  "Model",
  "Neurons",
  "Type",
  "Input",
  "Incoming",
  "Outgoing",
  "Problems",
] as const;

/**
 * Return the messages recorded against one object, or none.
 *
 * @param grouped - Located failures, already grouped by object identifier.
 * @param id - The population or projection to read them for.
 * @returns Its messages in the order the server reported them; empty when it
 *   was not refused.
 */
function messagesOf(
  grouped: ReadonlyMap<string, StudioGraphIssueLocation[]>,
  id: string,
): string[] {
  return (grouped.get(id) ?? []).map((location) => location.message);
}

/**
 * Return one entry per projection at one end of a population.
 *
 * The entry names the population at the *other* end, because that is what a
 * reader of the row needs: the row already says which population it is about.
 *
 * @param projections - Every projection in the graph.
 * @param labels - Population labels by identifier, for naming the far end.
 * @param grouped - Located failures by object identifier.
 * @param populationId - The population whose row is being built.
 * @param end - Which end of the projection this population is.
 * @returns One entry per incident projection, in graph order.
 */
function connectionsOf(
  projections: readonly ProjectionEdge[],
  labels: ReadonlyMap<string, string>,
  grouped: ReadonlyMap<string, StudioGraphIssueLocation[]>,
  populationId: string,
  end: "source" | "target",
): StudioGraphTableConnection[] {
  const other = end === "source" ? "target" : "source";
  return projections
    .filter((projection) => projection[end] === populationId)
    .map((projection) => ({
      detail: studioProjectionLabel(projection),
      id: projection.id,
      issues: messagesOf(grouped, projection.id),
      populationLabel: labels.get(projection[other]) ?? projection[other],
    }));
}

/**
 * Render connections as one sentence fragment, for the row's description.
 *
 * @param connections - The connections at one end of a population.
 * @returns The connections as one sentence fragment.
 */
function listed(connections: readonly StudioGraphTableConnection[]): string {
  return connections
    .map((connection) => `${connection.populationLabel} (${connection.detail})`)
    .join(", ");
}

/**
 * Return the single sentence a screen reader can read instead of the cells.
 *
 * It carries what the row carries, in the order the columns carry it, so a
 * reader who takes the sentence loses nothing by not walking the row.
 *
 * @param row - The row being described.
 * @returns One sentence carrying everything the row carries.
 */
function describe(row: Omit<StudioGraphTableRow, "description">): string {
  const identity = `${row.label}: ${row.count} ${row.neuronType} ${row.model} neurons, ${row.drive}`;
  const incoming =
    row.incoming.length === 0
      ? "no incoming projections"
      : `${row.incoming.length} incoming from ${listed(row.incoming)}`;
  const outgoing =
    row.outgoing.length === 0
      ? "no outgoing projections"
      : `${row.outgoing.length} outgoing to ${listed(row.outgoing)}`;
  const problems =
    row.issues.length === 0
      ? ""
      : ` Validation refused it: ${row.issues.join("; ")}`;
  return `${identity}; ${incoming}; ${outgoing}.${problems}`;
}

/**
 * Return the graph as a table.
 *
 * @param populations - Populations in the order the graph holds them.
 * @param projections - Projections in the order the graph holds them.
 * @param issues - Located validation failures, so a row can say what was
 *   refused about the object it describes rather than leaving the reader to
 *   match a flat list of sentences against the diagram.
 * @returns One row per population plus a caption stating the topology's size.
 */
export function studioGraphTable(
  populations: readonly PopulationNode[],
  projections: readonly ProjectionEdge[],
  issues: readonly StudioGraphIssueLocation[] = [],
): StudioGraphTable {
  const labels = new Map(populations.map((population) => [population.id, population.label]));
  const grouped = studioGraphIssuesById(issues);
  const rows = populations.map((population) => {
    const partial = {
      count: population.count,
      drive: studioPopulationDriveLabel(population.drive),
      id: population.id,
      incoming: connectionsOf(projections, labels, grouped, population.id, "target"),
      issues: messagesOf(grouped, population.id),
      label: population.label,
      model: population.model,
      neuronType: population.neuron_type,
      outgoing: connectionsOf(projections, labels, grouped, population.id, "source"),
    };
    return { ...partial, description: describe(partial) };
  });
  return { caption: studioGraphTableCaption(populations, projections), rows };
}

/**
 * Return the sentence that introduces the table.
 *
 * It states the size of the topology so a reader knows what the table holds
 * before entering it, and says plainly when the graph is empty.
 *
 * @param populations - Populations in the order the graph holds them.
 * @param projections - Projections in the order the graph holds them.
 * @returns The sentence that introduces the table.
 */
export function studioGraphTableCaption(
  populations: readonly PopulationNode[],
  projections: readonly ProjectionEdge[],
): string {
  if (populations.length === 0) {
    return "Network topology: no populations yet.";
  }
  const neurons = populations.reduce((total, population) => total + population.count, 0);
  const populationWord = populations.length === 1 ? "population" : "populations";
  const projectionWord = projections.length === 1 ? "projection" : "projections";
  const neuronWord = neurons === 1 ? "neuron" : "neurons";
  return (
    `Network topology: ${populations.length} ${populationWord} holding ` +
    `${neurons} ${neuronWord}, connected by ${projections.length} ${projectionWord}.`
  );
}

/**
 * Return the accessible name of the control that deletes one population.
 *
 * A row of buttons all called "Delete" is unusable without sight; each says
 * what it removes, including the projections that leave with it.
 *
 * @param row - The row being described.
 * @returns The accessible name of that row's delete control.
 */
export function studioGraphTableRemoveLabel(row: StudioGraphTableRow): string {
  const incident = row.incoming.length + row.outgoing.length;
  if (incident === 0) {
    return `Delete population ${row.label}`;
  }
  const projectionWord = incident === 1 ? "projection" : "projections";
  return `Delete population ${row.label} and its ${incident} ${projectionWord}`;
}

/**
 * Return how the table names one projection, from the row it leaves.
 *
 * Its controls sit beside every other projection's, so each names both ends
 * and what the projection carries: two projections between the same pair
 * differ in their detail, and a name without it would not tell them apart.
 *
 * @param row - The row of the projection's source population.
 * @param connection - The projection, as that row's outgoing connection.
 * @returns The projection's name, for its edit and delete controls.
 */
export function studioGraphTableProjectionName(
  row: StudioGraphTableRow,
  connection: StudioGraphTableConnection,
): string {
  return `projection ${row.label} to ${connection.populationLabel} (${connection.detail})`;
}
