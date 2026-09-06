// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Placing a graph validation failure on the object it names

/**
 * A refusal a reader cannot place is a refusal they have to hunt for.
 *
 * The server reports every failure at once with the request field it came
 * from. These cases hold the resolution of that field to the object it names,
 * and the two things that must never happen to a message: being reworded, and
 * being dropped because its field did not fit the shape this build expects.
 */

import { at } from "./arrayAt";
import { describe, expect, it } from "vitest";

import type { GraphValidation, PopulationNode, ProjectionEdge } from "./api/client";
import {
  studioGraphIssueLines,
  studioGraphIssueLocations,
  studioGraphIssuesById,
  studioGraphValidatedState,
  studioGraphValidationLocatedState,
} from "./studioGraphValidation";

/** One population carrying the label a located failure will name it by. */
function population(id: string, label: string): PopulationNode {
  return {
    count: 10,
    id,
    label,
    model: "SCLapicqueLIFNeuron",
    neuron_type: "excitatory",
    params: {},
    position: { x: 0, y: 0 },
    type: "population",
  };
}

const POPULATIONS = [population("p1", "Exc 0"), population("p2", "Inh 0")];
const PROJECTIONS: ProjectionEdge[] = [
  { delay: 0, id: "e1", probability: 0.1, rule: "random", source: "p1", target: "p2", weight: 0.5 },
];

const WEIGHT_ISSUE = {
  field: "projections[0].weight",
  message:
    "Projection e1 weight -4 conflicts with the excitatory source population p1; "
    + "excitatory sources need a positive weight",
};
const COUNT_ISSUE = {
  field: "populations[1].count",
  message: "Population Inh 0 count must be a positive integer",
};
const DT_ISSUE = { field: "dt", message: "Graph dt must be a positive finite number" };

describe("locating a failure", () => {
  it("resolves a population index to that population", () => {
    const located = at(studioGraphIssueLocations([COUNT_ISSUE], POPULATIONS, PROJECTIONS), 0);

    expect(located.kind).toBe("population");
    expect(located.id).toBe("p2");
    expect(located.subject).toBe("Inh 0");
    expect(located.attribute).toBe("count");
    expect(located.message).toBe(COUNT_ISSUE.message);
  });

  it("names a projection by its endpoints, which is how the canvas shows it", () => {
    const located = at(studioGraphIssueLocations([WEIGHT_ISSUE], POPULATIONS, PROJECTIONS), 0);

    expect(located.kind).toBe("projection");
    expect(located.id).toBe("e1");
    expect(located.subject).toBe("Exc 0 → Inh 0");
    expect(located.attribute).toBe("weight");
  });

  it("falls back to the identifier when an endpoint has no label to give", () => {
    const located = at(studioGraphIssueLocations([WEIGHT_ISSUE], [], PROJECTIONS), 0);

    expect(located.subject).toBe("p1 → p2");
  });

  it("keeps a failure about the run as a whole", () => {
    const located = at(studioGraphIssueLocations([DT_ISSUE], POPULATIONS, PROJECTIONS), 0);

    expect(located.kind).toBe("graph");
    expect(located.id).toBeNull();
    expect(located.subject).toBe("dt");
    expect(located.attribute).toBe("");
  });

  it("carries a nested attribute path whole", () => {
    const located = at(
      studioGraphIssueLocations(
        [{ field: "populations[0].params.tau", message: "unknown parameter tau" }],
        POPULATIONS,
        PROJECTIONS,
      ),
      0,
    );

    expect(located.id).toBe("p1");
    expect(located.attribute).toBe("params.tau");
  });

  it("keeps an index the graph no longer holds instead of dropping the message", () => {
    // The graph can change between the request and the answer. A message that
    // cannot be placed still has to be read.
    const located = at(
      studioGraphIssueLocations(
        [{ field: "projections[7].delay", message: "Projection e8 delay is not a whole step" }],
        POPULATIONS,
        PROJECTIONS,
      ),
      0,
    );

    expect(located.kind).toBe("graph");
    expect(located.id).toBeNull();
    expect(located.subject).toBe("projections[7].delay");
    expect(located.message).toBe("Projection e8 delay is not a whole step");
  });

  it("keeps a field shape this build does not recognise", () => {
    const located = at(
      studioGraphIssueLocations(
        [{ field: "monitors[0].kind", message: "unsupported monitor" }],
        POPULATIONS,
        PROJECTIONS,
      ),
      0,
    );

    expect(located.kind).toBe("graph");
    expect(located.message).toBe("unsupported monitor");
  });

  it("keeps the server's order, because that is the order the fields were read", () => {
    const located = studioGraphIssueLocations(
      [WEIGHT_ISSUE, COUNT_ISSUE, DT_ISSUE],
      POPULATIONS,
      PROJECTIONS,
    );

    expect(located.map((one) => one.field)).toEqual([
      "projections[0].weight",
      "populations[1].count",
      "dt",
    ]);
  });
});

describe("grouping located failures", () => {
  it("collects every failure of one object under its identifier", () => {
    const located = studioGraphIssueLocations(
      [
        WEIGHT_ISSUE,
        { field: "projections[0].delay", message: "Projection e1 delay is not a whole step" },
        COUNT_ISSUE,
      ],
      POPULATIONS,
      PROJECTIONS,
    );

    const grouped = studioGraphIssuesById(located);

    expect([...grouped.keys()].sort()).toEqual(["e1", "p2"]);
    expect(grouped.get("e1")).toHaveLength(2);
  });

  it("leaves failures about the run as a whole out of the map", () => {
    const located = studioGraphIssueLocations([DT_ISSUE], POPULATIONS, PROJECTIONS);

    expect(studioGraphIssuesById(located).size).toBe(0);
  });
});

describe("the lines a reader sees", () => {
  it("names the object and keeps the server's sentence verbatim", () => {
    const located = studioGraphIssueLocations([WEIGHT_ISSUE], POPULATIONS, PROJECTIONS);

    expect(studioGraphIssueLines(located)).toEqual([
      `Exc 0 → Inh 0: ${WEIGHT_ISSUE.message}`,
    ]);
  });

  it("does not prefix a failure whose only subject is its own field", () => {
    const located = studioGraphIssueLocations([DT_ISSUE], POPULATIONS, PROJECTIONS);

    expect(studioGraphIssueLines(located)).toEqual([DT_ISSUE.message]);
  });
});

describe("the state a refused validation leaves", () => {
  const validation: GraphValidation = {
    errors: [WEIGHT_ISSUE.message, COUNT_ISSUE.message],
    issues: [WEIGHT_ISSUE, COUNT_ISSUE],
    valid: false,
  };

  it("carries both the sentences and the located failures, and stops the run", () => {
    const patch = studioGraphValidationLocatedState(validation, POPULATIONS, PROJECTIONS);

    expect(patch.isSimulating).toBe(false);
    expect(patch.graphIssues.map((one) => one.id)).toEqual(["e1", "p2"]);
    expect(patch.graphErrors).toEqual([
      `Exc 0 → Inh 0: ${WEIGHT_ISSUE.message}`,
      `Inh 0: ${COUNT_ISSUE.message}`,
    ]);
  });

  it("clears a previous refusal when the server accepts the graph", () => {
    const patch = studioGraphValidatedState(
      { errors: [], issues: [], valid: true },
      POPULATIONS,
      PROJECTIONS,
    );

    expect(patch).toEqual({ graphErrors: [], graphIssues: [] });
  });

  it("says plainly when a refusal arrives with no reason at all", () => {
    // Saying "no errors" for a graph that was not accepted would read as
    // success. This is the answer a malformed or truncated reply deserves.
    const patch = studioGraphValidatedState(
      {} as unknown as GraphValidation,
      POPULATIONS,
      PROJECTIONS,
    );

    expect(patch.graphErrors).toEqual(["The server refused the graph without saying why."]);
    expect(patch.graphIssues).toEqual([]);
  });

  it("falls back to the server's own messages when it reported no fields", () => {
    // An older server, or one this build has not been upgraded alongside.
    const patch = studioGraphValidationLocatedState(
      { errors: ["something is wrong"], issues: [], valid: false },
      POPULATIONS,
      PROJECTIONS,
    );

    expect(patch.graphErrors).toEqual(["something is wrong"]);
    expect(patch.graphIssues).toEqual([]);
  });
});
