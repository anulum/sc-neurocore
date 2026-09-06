// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Where a graph validation failure belongs

/**
 * A validation message is only actionable once you know what it is about.
 *
 * The server reports every failure at once, which is right: a user should not
 * fix one field, resubmit, and discover the next. It also reports the request
 * field each message came from — `projections[2].delay`,
 * `populations[0].params.tau` — and that half used to be dropped on the floor.
 * A flat list of sentences leaves the reader matching prose against a diagram
 * to find which of six projections is meant.
 *
 * This resolves each field to the object it names, so a message can be shown
 * beside that object. An index the graph no longer holds, or a field shape
 * this build does not recognise, is kept as a graph-level issue rather than
 * discarded: a message that cannot be placed still has to be read.
 */

import type { GraphValidation, PopulationNode, ProjectionEdge } from "./api/client";

/** One failure exactly as the server reported it. */
export interface StudioGraphIssue {
  field: string;
  message: string;
}

/** What kind of object a failure belongs to. */
export type StudioGraphIssueKind = "population" | "projection" | "graph";

/** One failure, resolved to the object it is about. */
export interface StudioGraphIssueLocation {
  /** The object kind, or `graph` for a failure about the run as a whole. */
  kind: StudioGraphIssueKind;
  /** Identifier of that object; `null` when the failure is not about one. */
  id: string | null;
  /** How the object is named to a reader: a label, or an endpoint pair. */
  subject: string;
  /** The attribute inside the object, `delay` or `params.tau`; may be empty. */
  attribute: string;
  field: string;
  message: string;
}

const INDEXED = /^(populations|projections)\[(\d+)\](?:\.(.+))?$/;

/**
 * Name a projection the way the canvas draws it: by its endpoints.
 *
 * An identifier means nothing to a reader looking at a diagram; the labels of
 * the two populations it joins are how the projection is recognised.
 *
 * @param projection - The projection to name.
 * @param labels - Population labels by identifier.
 * @returns The two endpoint labels joined by an arrow, falling back to the identifiers when a label is not there to give.
 */
function projectionSubject(
  projection: ProjectionEdge,
  labels: ReadonlyMap<string, string>,
): string {
  const source = labels.get(projection.source) ?? projection.source;
  const target = labels.get(projection.target) ?? projection.target;
  return `${source} → ${target}`;
}

/**
 * Resolve every reported failure to the object its field names.
 *
 * @param issues - Failures as the server reported them, in its own order.
 * @param populations - Populations in the order they were sent for validation.
 * @param projections - Projections in the order they were sent for validation.
 * @returns One location per issue, in the order the server reported them.
 */
export function studioGraphIssueLocations(
  issues: readonly StudioGraphIssue[],
  populations: readonly PopulationNode[],
  projections: readonly ProjectionEdge[],
): StudioGraphIssueLocation[] {
  const labels = new Map(populations.map((population) => [population.id, population.label]));
  return issues.map((issue) => {
    const matched = INDEXED.exec(issue.field);
    const attribute = matched?.[3] ?? "";
    if (matched?.[1] === "populations") {
      const population = populations[Number(matched[2])];
      // An index the graph no longer holds is `undefined` at runtime; the
      // array type says otherwise only because `noUncheckedIndexedAccess`
      // is not on yet, and this guard is what keeps an unplaceable message
      // from being dropped.
      // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
      if (population !== undefined) {
        return {
          attribute,
          field: issue.field,
          id: population.id,
          kind: "population",
          message: issue.message,
          subject: population.label,
        };
      }
    }
    if (matched?.[1] === "projections") {
      const projection = projections[Number(matched[2])];
      // An index the graph no longer holds is `undefined` at runtime; the
      // array type says otherwise only because `noUncheckedIndexedAccess`
      // is not on yet, and this guard is what keeps an unplaceable message
      // from being dropped.
      // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
      if (projection !== undefined) {
        return {
          attribute,
          field: issue.field,
          id: projection.id,
          kind: "projection",
          message: issue.message,
          subject: projectionSubject(projection, labels),
        };
      }
    }
    // An index the graph no longer holds, or a field this build does not
    // recognise. The message is still shown, named by its own field.
    return {
      attribute,
      field: issue.field,
      id: null,
      kind: "graph",
      message: issue.message,
      subject: issue.field,
    };
  });
}

/**
 * Group located failures by the identifier of the object they are about.
 *
 * Graph-level failures are not in the map; read them from the located list.
 *
 * @param locations - Located failures for the whole graph.
 * @returns The failures of each object, by that object's identifier.
 */
export function studioGraphIssuesById(
  locations: readonly StudioGraphIssueLocation[],
): Map<string, StudioGraphIssueLocation[]> {
  const grouped = new Map<string, StudioGraphIssueLocation[]>();
  for (const location of locations) {
    if (location.id === null) continue;
    const existing = grouped.get(location.id);
    if (existing === undefined) {
      grouped.set(location.id, [location]);
    } else {
      existing.push(location);
    }
  }
  return grouped;
}

/**
 * Return one line per failure, each naming the object it is about.
 *
 * The message the server wrote is kept verbatim; only the subject is added,
 * because a message that has been reworded can no longer be matched against
 * what the server said.
 *
 * @param locations - Located failures for the whole graph.
 * @returns One line per failure, each naming the object it is about.
 */
export function studioGraphIssueLines(
  locations: readonly StudioGraphIssueLocation[],
): string[] {
  return locations.map((location) =>
    location.kind === "graph" && location.subject === location.field
      ? location.message
      : `${location.subject}: ${location.message}`,
  );
}

/**
 * Return what a validation answer says about the graph.
 *
 * Both forms are kept: the sentences the server wrote, and the same failures
 * resolved to the objects they name. A graph the server accepted clears both,
 * so a stale refusal never outlives the edit that fixed it.
 *
 * @param validation - The server's answer to a validation request.
 * @param populations - Populations in the order they were sent for validation.
 * @param projections - Projections in the order they were sent for validation.
 * @returns The messages and the located failures to put on the state.
 */
export function studioGraphValidatedState(
  validation: GraphValidation,
  populations: readonly PopulationNode[],
  projections: readonly ProjectionEdge[],
): { graphErrors: string[]; graphIssues: StudioGraphIssueLocation[] } {
  if (validation.valid) {
    return { graphErrors: [], graphIssues: [] };
  }
  const graphIssues = studioGraphIssueLocations(
    Array.isArray(validation.issues) ? validation.issues : [],
    populations,
    projections,
  );
  if (graphIssues.length > 0) {
    return { graphErrors: studioGraphIssueLines(graphIssues), graphIssues };
  }
  if (Array.isArray(validation.errors) && validation.errors.length > 0) {
    return { graphErrors: validation.errors, graphIssues };
  }
  // The graph was not accepted and no reason came back. Saying "no errors"
  // here would read as success; saying so plainly is the only honest answer.
  return {
    graphErrors: ["The server refused the graph without saying why."],
    graphIssues,
  };
}

/**
 * Return the state a refused validation leaves behind before a run.
 *
 * The run stops, so the busy flag is cleared here rather than by the caller.
 *
 * @param validation - The server's answer to a validation request.
 * @param populations - Populations in the order they were sent for validation.
 * @param projections - Projections in the order they were sent for validation.
 * @returns The same, with the run stopped.
 */
export function studioGraphValidationLocatedState(
  validation: GraphValidation,
  populations: readonly PopulationNode[],
  projections: readonly ProjectionEdge[],
): { graphErrors: string[]; graphIssues: StudioGraphIssueLocation[]; isSimulating: false } {
  return {
    ...studioGraphValidatedState(validation, populations, projections),
    isSimulating: false,
  };
}
