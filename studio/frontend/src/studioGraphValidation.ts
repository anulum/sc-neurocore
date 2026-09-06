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
 * Return the state a refused validation leaves behind.
 *
 * Both forms are kept: the sentences the server wrote, and the same failures
 * resolved to the objects they name. The run stops, so the busy flag is
 * cleared here rather than by the caller.
 */
export function studioGraphValidationLocatedState(
  validation: GraphValidation,
  populations: readonly PopulationNode[],
  projections: readonly ProjectionEdge[],
): { graphErrors: string[]; graphIssues: StudioGraphIssueLocation[]; isSimulating: false } {
  const graphIssues = studioGraphIssueLocations(
    validation.issues ?? [],
    populations,
    projections,
  );
  return {
    graphErrors:
      graphIssues.length > 0 ? studioGraphIssueLines(graphIssues) : validation.errors,
    graphIssues,
    isSimulating: false,
  };
}
