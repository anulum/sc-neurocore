// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Duplicating a selection, and what it refuses to guess

/**
 * The whole operation turns on one question, and these cases answer it.
 *
 * A projection with one end inside the selection and one end outside has no
 * correct copy: pointing it at the original preserves a fan-in, pointing it at
 * the duplicate preserves a motif, and either is silently wrong for the other
 * reader. So it is not copied, and the count of those left behind is carried
 * out to the user rather than left to be discovered in the diagram.
 *
 * The rest hold what a copy must carry — every executed field — and what it
 * must not do: invent an identifier, or produce a label indistinguishable from
 * the original.
 */

import { at } from "./arrayAt";
import { describe, expect, it } from "vitest";

import type { PopulationNode, ProjectionEdge } from "./api/client";
import {
  STUDIO_DUPLICATE_OFFSET,
  studioCrossingProjections,
  studioDuplicateLabel,
  studioDuplicatePlan,
  studioDuplicateSummary,
  studioInternalProjections,
} from "./studioGraphDuplicate";

/** One population, overridden where a case needs a particular value. */
function population(overrides: Partial<PopulationNode> & { id: string }): PopulationNode {
  return {
    count: 80,
    drive: { kind: "none" },
    label: overrides.id,
    model: "SCLapicqueLIFNeuron",
    neuron_type: "excitatory",
    params: {},
    position: { x: 10, y: 20 },
    type: "population",
    ...overrides,
  };
}

/** One projection, overridden where a case needs a particular value. */
function projection(overrides: Partial<ProjectionEdge> & { id: string }): ProjectionEdge {
  return {
    delay: 0,
    probability: 0.2,
    rule: "random",
    source: "a",
    target: "b",
    weight: 40,
    ...overrides,
  };
}

const A = population({ id: "a", label: "Exc 0" });
const B = population({ id: "b", label: "Inh 0", neuron_type: "inhibitory" });
const C = population({ id: "c", label: "Exc 1" });
const AB = projection({ id: "ab", source: "a", target: "b" });
const BC = projection({ id: "bc", source: "b", target: "c" });

describe("which projections a copy can carry", () => {
  it("carries the ones with both ends inside the selection", () => {
    const inside = studioInternalProjections([AB, BC], new Set(["a", "b"]));

    expect(inside.map((one) => one.id)).toEqual(["ab"]);
  });

  it("leaves the ones that cross the selection's boundary", () => {
    const crossing = studioCrossingProjections([AB, BC], new Set(["a", "b"]));

    expect(crossing.map((one) => one.id)).toEqual(["bc"]);
  });

  it("treats a projection wholly outside the selection as neither", () => {
    const selected = new Set(["a"]);

    expect(studioInternalProjections([BC], selected)).toEqual([]);
    expect(studioCrossingProjections([BC], selected)).toEqual([]);
  });

  it("carries a self-projection, whose two ends are the same population", () => {
    const loop = projection({ id: "aa", source: "a", target: "a" });

    expect(studioInternalProjections([loop], new Set(["a"]))).toHaveLength(1);
    expect(studioCrossingProjections([loop], new Set(["a"]))).toHaveLength(0);
  });
});

describe("the label a copy carries", () => {
  it("is distinguishable from the original", () => {
    expect(studioDuplicateLabel("Exc 0", ["Exc 0"])).toBe("Exc 0 copy");
  });

  it("keeps counting when the obvious name is taken", () => {
    expect(studioDuplicateLabel("Exc 0", ["Exc 0", "Exc 0 copy"])).toBe("Exc 0 copy 2");
    expect(studioDuplicateLabel("Exc 0", ["Exc 0", "Exc 0 copy", "Exc 0 copy 2"])).toBe(
      "Exc 0 copy 3",
    );
  });

  it("gives two copies of one label two different labels", () => {
    // Both populations are called "Shared"; one plan must not name both copies
    // the same thing.
    const shared = [population({ id: "x", label: "Shared" }), population({ id: "y", label: "Shared" })];

    const plan = studioDuplicatePlan(shared, [], ["x", "y"]);

    expect(new Set(plan.populations.map((one) => one.label)).size).toBe(2);
  });
});

describe("the plan for one selection", () => {
  it("copies every executed field of a population", () => {
    const source = population({
      count: 12,
      drive: { current: 1.5, kind: "constant" },
      id: "a",
      label: "Exc 0",
      model: "AdExNeuron",
      neuron_type: "inhibitory",
      params: { tau: 11 },
    });

    const copy = at(studioDuplicatePlan([source], [], ["a"]).populations, 0);

    expect(copy.count).toBe(12);
    expect(copy.model).toBe("AdExNeuron");
    expect(copy.neuron_type).toBe("inhibitory");
    expect(copy.params).toEqual({ tau: 11 });
    expect(copy.drive).toEqual({ current: 1.5, kind: "constant" });
    expect(copy.sourceId).toBe("a");
  });

  it("copies the parameters rather than sharing the object", () => {
    // A shared object would let an edit to the copy reach the original.
    const source = population({ id: "a", params: { tau: 11 } });

    const copy = at(studioDuplicatePlan([source], [], ["a"]).populations, 0);
    copy.params.tau = 99;

    expect(source.params.tau).toBe(11);
  });

  it("offsets the copy so it is visible as one", () => {
    const copy = at(studioDuplicatePlan([A], [], ["a"]).populations, 0);

    expect(copy.x).toBe(A.position.x + STUDIO_DUPLICATE_OFFSET.x);
    expect(copy.y).toBe(A.position.y + STUDIO_DUPLICATE_OFFSET.y);
  });

  it("copies every executed field of an internal projection", () => {
    const edge = projection({
      autapses: true,
      delay: 2,
      id: "ab",
      probability: 0.35,
      rule: "random",
      seed: 7,
      source: "a",
      target: "b",
      weight: -3.5,
    });

    const copy = at(studioDuplicatePlan([A, B], [edge], ["a", "b"]).projections, 0);

    expect(copy).toEqual({
      autapses: true,
      delay: 2,
      probability: 0.35,
      rule: "random",
      seed: 7,
      sourceId: "a",
      targetId: "b",
      weight: -3.5,
    });
  });

  it("treats a projection saved before rules existed as random", () => {
    const edge = projection({ id: "ab", probability: undefined, rule: undefined });

    const copy = at(studioDuplicatePlan([A, B], [edge], ["a", "b"]).projections, 0);

    expect(copy.rule).toBe("random");
  });

  it("counts what it left at the boundary", () => {
    const plan = studioDuplicatePlan([A, B, C], [AB, BC], ["a", "b"]);

    expect(plan.populations).toHaveLength(2);
    expect(plan.projections).toHaveLength(1);
    expect(plan.crossing).toBe(1);
  });

  it("ignores a selected identifier the graph no longer holds", () => {
    // A selection can outlive the population it named.
    const plan = studioDuplicatePlan([A], [], ["a", "gone"]);

    expect(plan.populations.map((one) => one.sourceId)).toEqual(["a"]);
  });

  it("plans nothing for an empty selection", () => {
    const plan = studioDuplicatePlan([A, B], [AB], []);

    expect(plan).toEqual({ crossing: 0, populations: [], projections: [] });
  });
});

describe("what the user is told", () => {
  it("says what was copied", () => {
    const plan = studioDuplicatePlan([A, B], [AB], ["a", "b"]);

    expect(studioDuplicateSummary(plan)).toBe(
      "Duplicated 2 populations and 1 projection between them.",
    );
  });

  it("says what was left behind, and why", () => {
    const plan = studioDuplicatePlan([A, B, C], [AB, BC], ["a", "b"]);

    const summary = studioDuplicateSummary(plan);

    expect(summary).toContain("1 projection left the selection and were not copied");
    expect(summary).toContain("cannot say whether it should reach the original or the duplicate");
  });

  it("uses the singular for one population", () => {
    expect(studioDuplicateSummary(studioDuplicatePlan([A], [], ["a"]))).toBe(
      "Duplicated 1 population and 0 projections between them.",
    );
  });

  it("says nothing at all when nothing was copied", () => {
    expect(studioDuplicateSummary(studioDuplicatePlan([A], [], []))).toBe("");
  });
});
