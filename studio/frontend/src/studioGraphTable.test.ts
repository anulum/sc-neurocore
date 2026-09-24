// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — The graph's topology stated as a table

/**
 * A reader who cannot see the canvas still learns what is connected to what.
 *
 * The canvas carries its meaning in positions and arrows, and neither survives
 * a screen reader. These cases pin the text that replaces them: the caption
 * that states the size of the topology before a reader enters the table, the
 * sentence each row carries, and the accessible name of a delete control that
 * says what leaves with the population.
 *
 * The table is a second presentation of one graph, so every string here is
 * derived from the same fields the canvas draws and the server runs.
 */

import { at } from "./arrayAt";
import { describe, expect, it } from "vitest";

import type { PopulationNode, ProjectionEdge } from "./api/client";
import {
  STUDIO_GRAPH_TABLE_COLUMNS,
  studioGraphTable,
  studioGraphTableCaption,
  studioGraphTableProjectionName,
  studioGraphTableRemoveLabel,
} from "./studioGraphTable";

/** One population, overridden where a case needs a particular value. */
function population(overrides: Partial<PopulationNode> & { id: string }): PopulationNode {
  return {
    count: 100,
    label: overrides.id,
    model: "LIF",
    neuron_type: "excitatory",
    params: {},
    position: { x: 0, y: 0 },
    type: "population",
    ...overrides,
  };
}

/** One projection, overridden where a case needs a particular value. */
function projection(overrides: Partial<ProjectionEdge> & { id: string }): ProjectionEdge {
  return {
    delay: 0,
    probability: 0.1,
    rule: "random",
    source: "input",
    target: "hidden",
    weight: 0.5,
    ...overrides,
  };
}

const input = population({
  drive: { current: 1.2, kind: "constant" },
  id: "input",
  label: "Input",
});
const hidden = population({
  count: 1,
  id: "hidden",
  label: "Hidden",
  model: "AdEx",
  neuron_type: "inhibitory",
});
const output = population({ count: 20, id: "output", label: "Output" });

const inputToHidden = projection({ id: "p1", source: "input", target: "hidden" });
const hiddenToOutput = projection({
  delay: 2,
  id: "p2",
  rule: "all_to_all",
  source: "hidden",
  target: "output",
  weight: -0.4,
});

describe("studioGraphTable", () => {
  it("gives every population a row carrying what reaches it and what it reaches", () => {
    const table = studioGraphTable([input, hidden, output], [inputToHidden, hiddenToOutput]);

    expect(table.rows.map((row) => row.id)).toEqual(["input", "hidden", "output"]);
    const first = at(table.rows, 0);
    const second = at(table.rows, 1);
    const third = at(table.rows, 2);
    expect(first.incoming).toEqual([]);
    expect(first.outgoing).toEqual([
      { detail: "w=0.5 p=0.1", id: "p1", issues: [], populationLabel: "Hidden" },
    ]);
    expect(second.incoming).toEqual([
      { detail: "w=0.5 p=0.1", id: "p1", issues: [], populationLabel: "Input" },
    ]);
    expect(second.outgoing).toEqual([
      { detail: "w=-0.4 all d=2ms", id: "p2", issues: [], populationLabel: "Output" },
    ]);
    expect(third.outgoing).toEqual([]);
  });

  it("carries the population's own identity in the row", () => {
    const row = at(studioGraphTable([input, hidden], [inputToHidden]).rows, 1);

    expect(row.label).toBe("Hidden");
    expect(row.model).toBe("AdEx");
    expect(row.count).toBe(1);
    expect(row.neuronType).toBe("inhibitory");
    expect(row.drive).toBe("no input");
  });

  it("describes a connected population in one sentence", () => {
    const row = at(studioGraphTable([input, hidden, output], [inputToHidden, hiddenToOutput]).rows, 1);

    expect(row.description).toBe(
      "Hidden: 1 inhibitory AdEx neurons, no input; " +
        "1 incoming from Input (w=0.5 p=0.1); " +
        "1 outgoing to Output (w=-0.4 all d=2ms).",
    );
  });

  it("says plainly when a population is connected to nothing", () => {
    const row = at(studioGraphTable([output], []).rows, 0);

    expect(row.description).toBe(
      "Output: 20 excitatory LIF neurons, no input; " +
        "no incoming projections; no outgoing projections.",
    );
  });

  it("names the drive the canvas names", () => {
    const row = at(studioGraphTable([input], []).rows, 0);

    expect(row.drive).toBe("I = 1.2");
    expect(row.description).toContain("I = 1.2");
  });

  it("falls back to the identifier when a projection names a population the graph lost", () => {
    const row = at(studioGraphTable([hidden], [inputToHidden]).rows, 0);

    expect(row.incoming).toEqual([
      { detail: "w=0.5 p=0.1", id: "p1", issues: [], populationLabel: "input" },
    ]);
  });

  it("lists the columns in the order the rows carry them", () => {
    expect(STUDIO_GRAPH_TABLE_COLUMNS).toEqual([
      "Population",
      "Model",
      "Neurons",
      "Type",
      "Input",
      "Incoming",
      "Outgoing",
      "Problems",
    ]);
  });
});

describe("studioGraphTableCaption", () => {
  it("states the size of the topology before a reader enters the table", () => {
    expect(studioGraphTableCaption([input, hidden, output], [inputToHidden, hiddenToOutput])).toBe(
      "Network topology: 3 populations holding 121 neurons, connected by 2 projections.",
    );
  });

  it("counts one of each in the singular", () => {
    expect(studioGraphTableCaption([hidden], [inputToHidden])).toBe(
      "Network topology: 1 population holding 1 neuron, connected by 1 projection.",
    );
  });

  it("says the graph is empty rather than reporting zeroes", () => {
    expect(studioGraphTableCaption([], [])).toBe("Network topology: no populations yet.");
  });

  it("introduces the table it captions", () => {
    const table = studioGraphTable([input, hidden], [inputToHidden]);

    expect(table.caption).toBe(studioGraphTableCaption([input, hidden], [inputToHidden]));
  });
});

describe("studioGraphTableRemoveLabel", () => {
  it("names the population a delete control removes", () => {
    const row = at(studioGraphTable([output], []).rows, 0);

    expect(studioGraphTableRemoveLabel(row)).toBe("Delete population Output");
  });

  it("warns that one incident projection leaves with the population", () => {
    const row = at(studioGraphTable([input, hidden], [inputToHidden]).rows, 0);

    expect(studioGraphTableRemoveLabel(row)).toBe("Delete population Input and its 1 projection");
  });

  it("counts projections at both ends of the population", () => {
    const row = at(
      studioGraphTable([input, hidden, output], [inputToHidden, hiddenToOutput]).rows,
      1,
    );

    expect(studioGraphTableRemoveLabel(row)).toBe("Delete population Hidden and its 2 projections");
  });
});

describe("studioGraphTableProjectionName", () => {
  it("names both ends and what the projection carries", () => {
    const row = at(studioGraphTable([hidden, output], [hiddenToOutput]).rows, 0);

    expect(studioGraphTableProjectionName(row, at(row.outgoing, 0))).toBe(
      "projection Hidden to Output (w=-0.4 all d=2ms)",
    );
  });

  it("tells apart two projections between the same pair", () => {
    const second = projection({ id: "p3", source: "input", target: "hidden", weight: 0.9 });
    const row = at(studioGraphTable([input, hidden], [inputToHidden, second]).rows, 0);

    const names = row.outgoing.map((connection) => studioGraphTableProjectionName(row, connection));
    expect(new Set(names).size).toBe(2);
  });
});

describe("a graph validation refused", () => {
  const ISSUES = [
    {
      attribute: "weight",
      field: "projections[0].weight",
      id: "p1",
      kind: "projection" as const,
      message: "Projection p1 weight -4 conflicts with the excitatory source population input",
      subject: "Input → Hidden",
    },
    {
      attribute: "count",
      field: "populations[1].count",
      id: "hidden",
      kind: "population" as const,
      message: "Population Hidden count must be a positive integer",
      subject: "Hidden",
    },
  ];

  it("puts a population's failure on that population's row", () => {
    const row = at(studioGraphTable([input, hidden], [inputToHidden], ISSUES).rows, 1);

    expect(row.issues).toEqual(["Population Hidden count must be a positive integer"]);
  });

  it("puts a projection's failure on that projection's connection entries", () => {
    const { rows } = studioGraphTable([input, hidden], [inputToHidden], ISSUES);
    const first = at(rows, 0);
    const second = at(rows, 1);

    expect(at(first.outgoing, 0).issues).toEqual([
      "Projection p1 weight -4 conflicts with the excitatory source population input",
    ]);
    expect(at(second.incoming, 0).issues).toEqual(at(first.outgoing, 0).issues);
    expect(first.issues).toEqual([]);
  });

  it("says in the row's sentence that the population was refused", () => {
    const row = at(studioGraphTable([input, hidden], [inputToHidden], ISSUES).rows, 1);

    expect(row.description).toContain(
      "Validation refused it: Population Hidden count must be a positive integer",
    );
  });

  it("leaves every row clean when nothing was refused", () => {
    const table = studioGraphTable([input, hidden], [inputToHidden]);

    expect(table.rows.every((row) => row.issues.length === 0)).toBe(true);
    expect(at(table.rows, 0).description).not.toContain("Validation refused");
  });
});
