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

/** One connection as a row states it. */
export interface StudioGraphTableConnection {
  /** Projection identifier, so a row can name what to remove. */
  id: string;
  /** The population at the other end. */
  populationLabel: string;
  /** Weight, rule and delay, in the canvas's own words. */
  detail: string;
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
] as const;

function connectionsOf(
  projections: readonly ProjectionEdge[],
  labels: ReadonlyMap<string, string>,
  populationId: string,
  end: "source" | "target",
): StudioGraphTableConnection[] {
  const other = end === "source" ? "target" : "source";
  return projections
    .filter((projection) => projection[end] === populationId)
    .map((projection) => ({
      detail: studioProjectionLabel(projection),
      id: projection.id,
      populationLabel: labels.get(projection[other]) ?? projection[other],
    }));
}

function listed(connections: readonly StudioGraphTableConnection[]): string {
  return connections
    .map((connection) => `${connection.populationLabel} (${connection.detail})`)
    .join(", ");
}

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
  return `${identity}; ${incoming}; ${outgoing}.`;
}

/**
 * Return the graph as a table.
 *
 * @param populations - Populations in the order the graph holds them.
 * @param projections - Projections in the order the graph holds them.
 * @returns One row per population plus a caption stating the topology's size.
 */
export function studioGraphTable(
  populations: readonly PopulationNode[],
  projections: readonly ProjectionEdge[],
): StudioGraphTable {
  const labels = new Map(populations.map((population) => [population.id, population.label]));
  const rows = populations.map((population) => {
    const partial = {
      count: population.count,
      drive: studioPopulationDriveLabel(population.drive),
      id: population.id,
      incoming: connectionsOf(projections, labels, population.id, "target"),
      label: population.label,
      model: population.model,
      neuronType: population.neuron_type,
      outgoing: connectionsOf(projections, labels, population.id, "source"),
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
 */
export function studioGraphTableRemoveLabel(row: StudioGraphTableRow): string {
  const incident = row.incoming.length + row.outgoing.length;
  if (incident === 0) {
    return `Delete population ${row.label}`;
  }
  const projectionWord = incident === 1 ? "projection" : "projections";
  return `Delete population ${row.label} and its ${incident} ${projectionWord}`;
}
