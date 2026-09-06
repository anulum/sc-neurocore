// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Copying part of a graph without inventing what it means

/**
 * Duplicating a selection, and being honest about what cannot come with it.
 *
 * Building the same motif twice is the commonest thing a user does with a
 * network editor, and the canvas made them do it by hand. The operation itself
 * is easy; what decides whether it is trustworthy is one question:
 *
 * **What happens to a projection with one end inside the selection and one
 * end outside?**
 *
 * There is no answer that is right for everyone. Pointing the copy at the
 * original preserves the fan-in a user may have been reproducing; pointing it
 * at the copy preserves the motif's shape. Either choice is silently wrong for
 * the other reader, and a wrong graph that ran is worse than one that refused.
 * So a crossing projection is **not copied**, and the plan says how many were
 * left behind, so the editor can tell the user rather than let them find out
 * by reading the diagram.
 *
 * Identity stays with the server. A duplicate's population and projection ids
 * are minted by the same routes that mint them for anything else — a client
 * that invented ids would be a second implementation of identity, free to
 * collide with the first.
 */

import type { PopulationNode, ProjectionEdge } from "./api/client";

/** How far a copy is offset from its original, so it is visible as a copy. */
export const STUDIO_DUPLICATE_OFFSET = { x: 40, y: 40 } as const;

/** One population to create, with the fields the copy carries over. */
export interface StudioDuplicatePopulation {
  /** The population being copied, so a caller can map old id to new. */
  sourceId: string;
  label: string;
  model: string;
  count: number;
  neuron_type: PopulationNode["neuron_type"];
  params: Record<string, number>;
  drive: PopulationNode["drive"];
  x: number;
  y: number;
}

/** One projection to create once both its endpoints have new ids. */
export interface StudioDuplicateProjection {
  sourceId: string;
  targetId: string;
  weight: number;
  delay: number;
  rule: NonNullable<ProjectionEdge["rule"]>;
  probability: number | undefined;
  /** Fields the creation route does not carry; applied after the id exists. */
  seed: number | undefined;
  autapses: boolean | undefined;
}

/** Everything one duplicate does, and everything it deliberately does not. */
export interface StudioDuplicatePlan {
  populations: StudioDuplicatePopulation[];
  projections: StudioDuplicateProjection[];
  /** Projections with exactly one end in the selection, which are not copied. */
  crossing: number;
}

/**
 * Return a label that is distinguishable from every label already in use.
 *
 * Two populations may legitimately share a label — the graph identifies them
 * by id — but a copy that reads identically to its original is a trap for the
 * person who has to tell them apart.
 *
 * @param label - The original's label.
 * @param taken - Labels already in the graph, including the original's.
 * @returns `"<label> copy"`, or `"<label> copy 2"` and upward when that is
 *   taken as well.
 */
export function studioDuplicateLabel(label: string, taken: readonly string[]): string {
  const used = new Set(taken);
  const first = `${label} copy`;
  if (!used.has(first)) return first;
  let index = 2;
  while (used.has(`${first} ${index}`)) index += 1;
  return `${first} ${index}`;
}

/**
 * Return the projections whose **both** endpoints are inside the selection.
 *
 * @param projections - Every projection in the graph.
 * @param selected - Identifiers of the selected populations.
 * @returns The projections a copy of that selection can carry unambiguously.
 */
export function studioInternalProjections(
  projections: readonly ProjectionEdge[],
  selected: ReadonlySet<string>,
): ProjectionEdge[] {
  return projections.filter(
    (projection) => selected.has(projection.source) && selected.has(projection.target),
  );
}

/**
 * Return the projections with exactly one end inside the selection.
 *
 * These are the ones a duplicate cannot carry, and the number of them is what
 * the editor tells the user.
 *
 * @param projections - Every projection in the graph.
 * @param selected - Identifiers of the selected populations.
 * @returns The projections that cross the selection's boundary.
 */
export function studioCrossingProjections(
  projections: readonly ProjectionEdge[],
  selected: ReadonlySet<string>,
): ProjectionEdge[] {
  return projections.filter(
    (projection) =>
      selected.has(projection.source) !== selected.has(projection.target),
  );
}

/**
 * Return the plan for duplicating one selection of populations.
 *
 * @param populations - Every population in the graph.
 * @param projections - Every projection in the graph.
 * @param selectedIds - Identifiers of the populations to copy; identifiers the
 *   graph does not hold are ignored, because a stale selection must not
 *   invent a population.
 * @returns The populations and projections to create, and how many projections
 *   were left behind at the selection's boundary.
 */
export function studioDuplicatePlan(
  populations: readonly PopulationNode[],
  projections: readonly ProjectionEdge[],
  selectedIds: readonly string[],
): StudioDuplicatePlan {
  const wanted = new Set(selectedIds);
  const chosen = populations.filter((population) => wanted.has(population.id));
  const selected = new Set(chosen.map((population) => population.id));
  const taken = populations.map((population) => population.label);
  const copies: StudioDuplicatePopulation[] = [];
  for (const population of chosen) {
    const label = studioDuplicateLabel(population.label, taken);
    // Each new label joins the taken set as it is minted, so two copies of
    // populations sharing a label do not both become "<label> copy".
    taken.push(label);
    copies.push({
      count: population.count,
      drive: population.drive,
      label,
      model: population.model,
      neuron_type: population.neuron_type,
      params: { ...population.params },
      sourceId: population.id,
      x: population.position.x + STUDIO_DUPLICATE_OFFSET.x,
      y: population.position.y + STUDIO_DUPLICATE_OFFSET.y,
    });
  }
  return {
    crossing: studioCrossingProjections(projections, selected).length,
    populations: copies,
    projections: studioInternalProjections(projections, selected).map((projection) => ({
      autapses: projection.autapses,
      delay: projection.delay,
      probability: projection.probability,
      rule: projection.rule ?? "random",
      seed: projection.seed,
      sourceId: projection.source,
      targetId: projection.target,
      weight: projection.weight,
    })),
  };
}

/**
 * Return what the editor tells the user a duplicate did.
 *
 * The count of projections left at the boundary is the part a user cannot see
 * from the diagram, so it is said in words rather than implied.
 *
 * @param plan - The plan that was carried out.
 * @returns One sentence, or an empty string when there was nothing to copy.
 */
export function studioDuplicateSummary(plan: StudioDuplicatePlan): string {
  if (plan.populations.length === 0) {
    return "";
  }
  const populationWord = plan.populations.length === 1 ? "population" : "populations";
  const copied =
    `Duplicated ${plan.populations.length} ${populationWord} ` +
    `and ${plan.projections.length} projection${plan.projections.length === 1 ? "" : "s"} between them.`;
  if (plan.crossing === 0) {
    return copied;
  }
  const crossingWord = plan.crossing === 1 ? "projection" : "projections";
  return (
    `${copied} ${plan.crossing} ${crossingWord} left the selection and ` +
    "were not copied: a copy of one cannot say whether it should reach the " +
    "original or the duplicate."
  );
}
