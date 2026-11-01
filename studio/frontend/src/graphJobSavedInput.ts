// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Saved editable graph input reader

import type { NetworkGraph, PopulationDrive, PopulationNode, ProjectionEdge } from "./api/types";

/**
 * Identify a named-field JSON container.
 *
 * @param value - Untrusted saved value.
 * @returns Whether named fields can be inspected.
 */
function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/**
 * Identify a finite editable scalar without interpreting its scientific admissibility.
 *
 * @param value - Untrusted saved field.
 * @returns Whether it is a finite number.
 */
function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/**
 * Check a saved input drive's structure, leaving model and work admission to the server.
 *
 * @param value - Untrusted saved drive, including legacy absence.
 * @returns Whether it has the declared editable drive shape.
 */
function drive(value: unknown): value is PopulationDrive | undefined {
  return value === undefined || record(value) && (value.kind === "none"
    || value.kind === "constant" && finite(value.current)
    || value.kind === "poisson" && finite(value.rate_hz) && finite(value.weight)
      && (value.seed === undefined || finite(value.seed)));
}

/**
 * Check the canvas fields needed to reopen one editable population safely.
 *
 * @param value - Untrusted saved node.
 * @returns Whether the UI can read its fields without substituting defaults.
 */
function population(value: unknown): value is PopulationNode {
  return record(value) && [value.id, value.label, value.model].every((field) => typeof field === "string")
    && value.type === "population" && finite(value.count)
    && (value.neuron_type === "excitatory" || value.neuron_type === "inhibitory")
    && record(value.position) && finite(value.position.x) && finite(value.position.y)
    && record(value.params) && Object.values(value.params).every(finite) && drive(value.drive);
}

/**
 * Check a projection without replacing a legacy omitted connectivity option.
 *
 * @param value - Untrusted saved edge.
 * @returns Whether every present editor field has its declared shape.
 */
function projection(value: unknown): value is ProjectionEdge {
  return record(value) && [value.id, value.source, value.target].every((field) => typeof field === "string")
    && finite(value.weight) && finite(value.delay)
    && (value.rule === undefined || value.rule === "random" || value.rule === "all_to_all")
    && (value.probability === undefined || finite(value.probability))
    && (value.seed === undefined || finite(value.seed))
    && (value.autapses === undefined || typeof value.autapses === "boolean");
}

/**
 * Read a saved graph's complete editable structure, preserving optional input fields.
 *
 * A structurally readable draft may still be scientifically invalid. Reopening
 * it retains that draft for correction; this reader never grants execution.
 *
 * @param value - Parsed saved graph or exact retained submission body.
 * @returns A detached graph with the original field values and no invented defaults.
 * @throws {Error} When explicit fields cannot be read as an editable graph.
 */
export function readGraphJobSavedInput(value: unknown): NetworkGraph {
  if (!record(value) || !Array.isArray(value.populations) || !Array.isArray(value.projections)
    || !value.populations.every(population) || !value.projections.every(projection)
    || value.duration !== undefined && !finite(value.duration) || value.dt !== undefined && !finite(value.dt)
    || value.seed !== undefined && !finite(value.seed)) throw new Error("Saved graph input is unreadable.");
  const graph: NetworkGraph = {
    populations: structuredClone(value.populations), projections: structuredClone(value.projections),
    ...(value.duration === undefined ? {} : { duration: value.duration }),
    ...(value.dt === undefined ? {} : { dt: value.dt }),
    ...(value.seed === undefined ? {} : { seed: value.seed }),
  };
  return graph;
}
