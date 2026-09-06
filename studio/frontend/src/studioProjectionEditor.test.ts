// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — The fields a projection is run with, made editable

/**
 * The canvas ran fields it could not edit.
 *
 * A projection's weight, rule, probability, delay, seed and autapse setting
 * are all executed by `studio.network-graph-spec.v1`, and every one of them
 * was fixed at the request-builder default the moment the edge was drawn.
 *
 * These cases hold the editing model to the one division that keeps it
 * honest: it parses, and the server decides. A value that is not a value is
 * refused here; a value that is not *admissible* goes to the server, because a
 * second copy of the contract in the browser is a copy free to drift from the
 * one that runs.
 */

import { describe, expect, it } from "vitest";

import type { PopulationNode, ProjectionEdge } from "./api/client";
import type { StudioGraphIssueLocation } from "./studioGraphValidation";
import {
  STUDIO_PROJECTION_FIELDS,
  STUDIO_PROJECTION_RULES,
  studioProjectionEdit,
  studioProjectionFieldErrors,
  studioProjectionFields,
  studioProjectionTitle,
} from "./studioProjectionEditor";

function population(id: string, label: string, inhibitory = false): PopulationNode {
  return {
    count: 10,
    id,
    label,
    model: "SCLapicqueLIFNeuron",
    neuron_type: inhibitory ? "inhibitory" : "excitatory",
    params: {},
    position: { x: 0, y: 0 },
    type: "population",
  };
}

const EXC = population("p1", "Exc 0");
const INH = population("p2", "Inh 0", true);

function projection(overrides: Partial<ProjectionEdge> = {}): ProjectionEdge {
  return {
    delay: 0,
    id: "e1",
    probability: 0.2,
    rule: "random",
    source: "p1",
    target: "p2",
    weight: 40,
    ...overrides,
  };
}

function fieldOf(models: ReturnType<typeof studioProjectionFields>, field: string) {
  const found = models.find((model) => model.field === field);
  if (found === undefined) throw new Error(`no field ${field}`);
  return found;
}

describe("the fields a projection offers", () => {
  it("offers every field the runtime executes, in a stable order", () => {
    const models = studioProjectionFields(projection(), EXC);

    expect(models.map((model) => model.field)).toEqual([...STUDIO_PROJECTION_FIELDS]);
  });

  it("carries the projection's current values", () => {
    const models = studioProjectionFields(
      projection({ autapses: true, delay: 2, seed: 7, weight: -3.5 }),
      INH,
    );

    expect(fieldOf(models, "weight").value).toBe("-3.5");
    expect(fieldOf(models, "delay").value).toBe("2");
    expect(fieldOf(models, "seed").value).toBe("7");
    expect(fieldOf(models, "autapses").checked).toBe(true);
  });

  it("leaves a derived seed empty rather than inventing one to show", () => {
    const models = studioProjectionFields(projection({ seed: undefined }), EXC);

    expect(fieldOf(models, "seed").value).toBe("");
  });

  it("states the sign the source population's type requires", () => {
    expect(fieldOf(studioProjectionFields(projection(), EXC), "weight").help).toContain(
      "excitatory source needs a positive weight",
    );
    expect(fieldOf(studioProjectionFields(projection(), INH), "weight").help).toContain(
      "inhibitory source needs a negative weight",
    );
  });

  it("states the constraint without a source rather than guessing one", () => {
    expect(fieldOf(studioProjectionFields(projection(), undefined), "weight").help).toContain(
      "must agree with the source population's type",
    );
  });

  it("marks probability as not applying under the all_to_all rule", () => {
    const models = studioProjectionFields(projection({ rule: "all_to_all" }), EXC);

    expect(fieldOf(models, "probability").applies).toBe(false);
    expect(fieldOf(models, "rule").applies).toBe(true);
  });

  it("treats a projection saved before rules existed as random", () => {
    const models = studioProjectionFields(projection({ rule: undefined }), EXC);

    expect(fieldOf(models, "rule").value).toBe("random");
    expect(fieldOf(models, "probability").applies).toBe(true);
  });

  it("offers exactly the rules the graph spec admits", () => {
    expect(fieldOf(studioProjectionFields(projection(), EXC), "rule").choices).toEqual([
      "random",
      "all_to_all",
    ]);
    expect(STUDIO_PROJECTION_RULES).toEqual(["random", "all_to_all"]);
  });
});

describe("a validation failure on a field", () => {
  const located: StudioGraphIssueLocation[] = [
    {
      attribute: "weight",
      field: "projections[0].weight",
      id: "e1",
      kind: "projection",
      message: "Projection e1 weight -4 conflicts with the excitatory source population p1",
      subject: "Exc 0 → Inh 0",
    },
    {
      attribute: "delay",
      field: "projections[0].delay",
      id: "e1",
      kind: "projection",
      message: "Projection e1 delay 0.05 ms is not a whole number of 0.1 ms steps",
      subject: "Exc 0 → Inh 0",
    },
    {
      attribute: "weight",
      field: "projections[1].weight",
      id: "e2",
      kind: "projection",
      message: "Projection e2 weight is not a finite number",
      subject: "Inh 0 → Exc 0",
    },
    {
      attribute: "count",
      field: "populations[0].count",
      id: "p1",
      kind: "population",
      message: "Population Exc 0 count must be a positive integer",
      subject: "Exc 0",
    },
  ];

  it("reaches the input the server named, and no other", () => {
    const models = studioProjectionFields(projection(), EXC, located);

    expect(fieldOf(models, "weight").errors).toEqual([
      "Projection e1 weight -4 conflicts with the excitatory source population p1",
    ]);
    expect(fieldOf(models, "delay").errors).toHaveLength(1);
    expect(fieldOf(models, "probability").errors).toEqual([]);
  });

  it("keeps another projection's failure off this projection", () => {
    const byField = studioProjectionFieldErrors(located, "e1");

    expect(byField.get("weight")).toHaveLength(1);
    expect(studioProjectionFieldErrors(located, "e2").get("weight")).toEqual([
      "Projection e2 weight is not a finite number",
    ]);
  });

  it("keeps a failure about the projection as a whole rather than losing it", () => {
    const whole: StudioGraphIssueLocation[] = [
      {
        attribute: "",
        field: "projections[0]",
        id: "e1",
        kind: "projection",
        message: "Projection e1 has unknown fields: colour",
        subject: "Exc 0 → Inh 0",
      },
    ];

    expect(studioProjectionFieldErrors(whole, "e1").get("")).toEqual([
      "Projection e1 has unknown fields: colour",
    ]);
  });
});

describe("turning a typed value into an edit", () => {
  it("parses a number", () => {
    expect(studioProjectionEdit("weight", "-3.5")).toEqual({
      ok: true,
      update: { weight: -3.5 },
    });
    expect(studioProjectionEdit("delay", " 2 ")).toEqual({ ok: true, update: { delay: 2 } });
  });

  it("refuses text where a number belongs, and says which field", () => {
    const refused = studioProjectionEdit("delay", "soon");

    expect(refused).toEqual({ ok: false, reason: "delay must be a number." });
  });

  it("refuses a blank required box rather than sending nothing", () => {
    expect(studioProjectionEdit("weight", "")).toEqual({
      ok: false,
      reason: "weight must be a number.",
    });
  });

  it("refuses a non-finite number, which no seal or spec would carry", () => {
    expect(studioProjectionEdit("weight", "Infinity")).toEqual({
      ok: false,
      reason: "weight must be a number.",
    });
    expect(studioProjectionEdit("weight", "NaN")).toEqual({
      ok: false,
      reason: "weight must be a number.",
    });
  });

  it("reads an empty seed as a request to derive one", () => {
    expect(studioProjectionEdit("seed", "  ")).toEqual({ ok: true, update: { seed: undefined } });
  });

  it("drops the probability when the rule stops using one", () => {
    // The spec refuses a probability under all_to_all, so sending one would be
    // an edit the server has to reject.
    expect(studioProjectionEdit("rule", "all_to_all")).toEqual({
      ok: true,
      update: { probability: undefined, rule: "all_to_all" },
    });
  });

  it("leaves the probability alone when the rule uses one", () => {
    expect(studioProjectionEdit("rule", "random")).toEqual({ ok: true, update: { rule: "random" } });
  });

  it("refuses a rule the graph spec does not admit", () => {
    expect(studioProjectionEdit("rule", "gaussian")).toEqual({
      ok: false,
      reason: "Rule must be one of random, all_to_all.",
    });
  });

  it("reads the autapse box both ways", () => {
    expect(studioProjectionEdit("autapses", true)).toEqual({
      ok: true,
      update: { autapses: true },
    });
    expect(studioProjectionEdit("autapses", false)).toEqual({
      ok: true,
      update: { autapses: false },
    });
  });

  it("does not decide admissibility, which belongs to the server", () => {
    // A weight whose sign contradicts its source is a real refusal — and it is
    // the server's, reported with its field. The editor must send it.
    expect(studioProjectionEdit("weight", "-40")).toEqual({
      ok: true,
      update: { weight: -40 },
    });
    expect(studioProjectionEdit("probability", "1.5")).toEqual({
      ok: true,
      update: { probability: 1.5 },
    });
    expect(studioProjectionEdit("delay", "0.05")).toEqual({ ok: true, update: { delay: 0.05 } });
  });
});

describe("naming the projection being edited", () => {
  it("names it by its endpoints, as the canvas draws it", () => {
    expect(studioProjectionTitle(projection(), [EXC, INH])).toBe("Exc 0 → Inh 0");
  });

  it("falls back to the identifier when an endpoint is gone", () => {
    expect(studioProjectionTitle(projection(), [])).toBe("p1 → p2");
  });
});
