// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — The fields a population is run with, made editable

/**
 * The canvas ran a population's model, count, type, drive and parameters
 * exactly as they were created, because it could not edit any of them.
 *
 * The parameters are the half that cannot be guessed: which constructor fields
 * a population may override, their kind and their default are decided by the
 * run contract and read from the server. These cases hold the model to that
 * contract, to the drive shapes the specification admits, and to the same
 * division as the projection editor — parse here, decide there.
 */

import { describe, expect, it } from "vitest";

import type { PopulationModelContract, PopulationNode } from "./api/client";
import type { StudioGraphIssueLocation } from "./studioGraphValidation";
import {
  STUDIO_DRIVE_KINDS,
  STUDIO_NEURON_TYPES,
  studioPopulationDriveFields,
  studioPopulationEdit,
  studioPopulationFieldErrors,
  studioPopulationFields,
  studioPopulationParameterFields,
  studioPopulationUnsupported,
} from "./studioPopulationEditor";

const MODELS = ["SCLapicqueLIFNeuron", "AdExNeuron"];

const CONTRACT: PopulationModelContract = {
  drive: { kind: "float", parameter: "current", positional_only: false },
  model: "SCLapicqueLIFNeuron",
  parameters: [
    { default: 1.1, kind: "float", name: "capacitance" },
    { default: null, kind: "float", name: "v" },
    { default: 3, kind: "int", name: "refractory_steps" },
  ],
  schema_version: "studio.population-model-contract.v1",
  unsupported: [
    { name: "dt", reason: "the timestep is set through the dt field, not a parameter override" },
    { name: "profile", reason: "non-numeric field" },
  ],
};

function population(overrides: Partial<PopulationNode> = {}): PopulationNode {
  return {
    count: 80,
    drive: { kind: "none" },
    id: "p1",
    label: "Exc 0",
    model: "SCLapicqueLIFNeuron",
    neuron_type: "excitatory",
    params: {},
    position: { x: 0, y: 0 },
    type: "population",
    ...overrides,
  };
}

function fieldOf(models: StudioPopulationFieldModelList, field: string) {
  const found = models.find((model) => model.field === field);
  if (found === undefined) throw new Error(`no field ${field}`);
  return found;
}

type StudioPopulationFieldModelList = ReturnType<typeof studioPopulationFields>;

describe("what a population offers", () => {
  it("offers the identity fields the graph executes", () => {
    const fields = studioPopulationFields(population(), MODELS);

    expect(fields.map((field) => field.field)).toEqual([
      "label",
      "model",
      "count",
      "neuron_type",
    ]);
  });

  it("offers only the models the server admits", () => {
    expect(fieldOf(studioPopulationFields(population(), MODELS), "model").choices).toEqual(MODELS);
  });

  it("says the type fixes the sign of every outgoing weight", () => {
    expect(fieldOf(studioPopulationFields(population(), MODELS), "neuron_type").help).toContain(
      "sign every outgoing projection",
    );
    expect(STUDIO_NEURON_TYPES).toEqual(["excitatory", "inhibitory"]);
  });

  it("carries the population's current values", () => {
    const fields = studioPopulationFields(population({ count: 5, label: "Sensory" }), MODELS);

    expect(fieldOf(fields, "label").value).toBe("Sensory");
    expect(fieldOf(fields, "count").value).toBe("5");
  });
});

describe("the external input", () => {
  it("offers only the kind when nothing drives the population", () => {
    const fields = studioPopulationDriveFields(population());

    expect(fields.map((field) => field.field)).toEqual(["drive.kind"]);
    expect(fields[0].choices).toEqual([...STUDIO_DRIVE_KINDS]);
  });

  it("offers the current of a constant drive", () => {
    const fields = studioPopulationDriveFields(
      population({ drive: { current: 1.2, kind: "constant" } }),
    );

    expect(fields.map((field) => field.field)).toEqual(["drive.kind", "drive.current"]);
    expect(fieldOf(fields, "drive.current").value).toBe("1.2");
  });

  it("offers the rate, weight and seed of a Poisson drive", () => {
    const fields = studioPopulationDriveFields(
      population({ drive: { kind: "poisson", rate_hz: 20, seed: 7, weight: 0.5 } }),
    );

    expect(fields.map((field) => field.field)).toEqual([
      "drive.kind",
      "drive.rate_hz",
      "drive.weight",
      "drive.seed",
    ]);
    expect(fieldOf(fields, "drive.seed").value).toBe("7");
  });

  it("leaves a derived Poisson seed empty rather than inventing one", () => {
    const fields = studioPopulationDriveFields(
      population({ drive: { kind: "poisson", rate_hz: 20, weight: 0.5 } }),
    );

    expect(fieldOf(fields, "drive.seed").value).toBe("");
  });

  it("treats a population saved before drives existed as undriven", () => {
    const fields = studioPopulationDriveFields(population({ drive: undefined }));

    expect(fieldOf(fields, "drive.kind").value).toBe("none");
  });
});

describe("the parameters the model contract declares", () => {
  it("offers one input per overridable parameter, in the contract's order", () => {
    const fields = studioPopulationParameterFields(population(), CONTRACT);

    expect(fields.map((field) => field.field)).toEqual([
      "params.capacitance",
      "params.v",
      "params.refractory_steps",
    ]);
  });

  it("shows the model's declared default for a parameter nothing overrides", () => {
    const fields = studioPopulationParameterFields(population(), CONTRACT);

    expect(fieldOf(fields, "params.capacitance").value).toBe("1.1");
    expect(fieldOf(fields, "params.capacitance").help).toContain("declares 1.1");
  });

  it("shows the override where there is one", () => {
    const fields = studioPopulationParameterFields(
      population({ params: { capacitance: 2.5 } }),
      CONTRACT,
    );

    expect(fieldOf(fields, "params.capacitance").value).toBe("2.5");
  });

  it("says so when a parameter declares no default at all", () => {
    const fields = studioPopulationParameterFields(population(), CONTRACT);

    expect(fieldOf(fields, "params.v").value).toBe("");
    expect(fieldOf(fields, "params.v").help).toContain("no default");
  });

  it("says a whole-number parameter is one", () => {
    expect(
      fieldOf(studioPopulationParameterFields(population(), CONTRACT), "params.refractory_steps")
        .help,
    ).toContain("Whole number");
  });

  it("offers nothing while the contract has not arrived", () => {
    expect(studioPopulationParameterFields(population(), null)).toEqual([]);
  });

  it("offers nothing when the contract is for a different model", () => {
    // Showing the previous model's parameters would offer fields the graph
    // refuses for this one.
    expect(
      studioPopulationParameterFields(population({ model: "AdExNeuron" }), CONTRACT),
    ).toEqual([]);
  });

  it("reports the fields that are not inputs, with the reason each is not", () => {
    expect(studioPopulationUnsupported(CONTRACT)).toEqual([
      {
        name: "dt",
        reason: "the timestep is set through the dt field, not a parameter override",
      },
      { name: "profile", reason: "non-numeric field" },
    ]);
    expect(studioPopulationUnsupported(null)).toEqual([]);
  });
});

describe("a validation failure on a field", () => {
  const located: StudioGraphIssueLocation[] = [
    {
      attribute: "count",
      field: "populations[0].count",
      id: "p1",
      kind: "population",
      message: "Population Exc 0 count must be a positive integer",
      subject: "Exc 0",
    },
    {
      attribute: "params.capacitance",
      field: "populations[0].params.capacitance",
      id: "p1",
      kind: "population",
      message: "Population Exc 0 params.capacitance: unknown parameter",
      subject: "Exc 0",
    },
    {
      attribute: "drive.current",
      field: "populations[0].drive.current",
      id: "p1",
      kind: "population",
      message: "Population Exc 0 constant drive needs a finite current",
      subject: "Exc 0",
    },
    {
      attribute: "count",
      field: "populations[1].count",
      id: "p2",
      kind: "population",
      message: "Population Inh 0 count must be a positive integer",
      subject: "Inh 0",
    },
  ];

  it("reaches the identity input the server named", () => {
    expect(fieldOf(studioPopulationFields(population(), MODELS, located), "count").errors).toEqual([
      "Population Exc 0 count must be a positive integer",
    ]);
  });

  it("reaches the parameter input the server named", () => {
    const fields = studioPopulationParameterFields(population(), CONTRACT, located);

    expect(fieldOf(fields, "params.capacitance").errors).toEqual([
      "Population Exc 0 params.capacitance: unknown parameter",
    ]);
    expect(fieldOf(fields, "params.v").errors).toEqual([]);
  });

  it("reaches the drive input the server named", () => {
    const fields = studioPopulationDriveFields(
      population({ drive: { current: 1.2, kind: "constant" } }),
      located,
    );

    expect(fieldOf(fields, "drive.current").errors).toHaveLength(1);
  });

  it("keeps another population's failure off this one", () => {
    expect(studioPopulationFieldErrors(located, "p2").get("count")).toEqual([
      "Population Inh 0 count must be a positive integer",
    ]);
    expect(studioPopulationFieldErrors(located, "p1").get("count")).toHaveLength(1);
  });
});

describe("turning a typed value into an edit", () => {
  it("renames a population", () => {
    expect(studioPopulationEdit(population(), "label", "Sensory")).toEqual({
      ok: true,
      update: { label: "Sensory" },
    });
  });

  it("refuses an empty label rather than an unnameable population", () => {
    expect(studioPopulationEdit(population(), "label", "  ")).toEqual({
      ok: false,
      reason: "label must not be empty.",
    });
  });

  it("clears the parameters when the model changes", () => {
    // The previous model's overrides mean nothing to the new one, and the
    // graph refuses a parameter the model does not declare.
    expect(
      studioPopulationEdit(population({ params: { capacitance: 2 } }), "model", "AdExNeuron"),
    ).toEqual({ ok: true, update: { model: "AdExNeuron", params: {} } });
  });

  it("changes the neuron type, and refuses one that is not a type", () => {
    expect(studioPopulationEdit(population(), "neuron_type", "inhibitory")).toEqual({
      ok: true,
      update: { neuron_type: "inhibitory" },
    });
    expect(studioPopulationEdit(population(), "neuron_type", "mixed")).toEqual({
      ok: false,
      reason: "neuron_type must be one of excitatory, inhibitory.",
    });
  });

  it("parses a count and refuses text", () => {
    expect(studioPopulationEdit(population(), "count", "40")).toEqual({
      ok: true,
      update: { count: 40 },
    });
    expect(studioPopulationEdit(population(), "count", "many")).toEqual({
      ok: false,
      reason: "count must be a number.",
    });
  });

  it("sends a count the server will refuse, because that answer is the server's", () => {
    expect(studioPopulationEdit(population(), "count", "0")).toEqual({
      ok: true,
      update: { count: 0 },
    });
    expect(studioPopulationEdit(population(), "count", "2.5")).toEqual({
      ok: true,
      update: { count: 2.5 },
    });
  });

  it("gives a new drive kind only the fields that kind carries", () => {
    // A drive of kind none that carries fields is refused by the graph.
    expect(studioPopulationEdit(population(), "drive.kind", "none")).toEqual({
      ok: true,
      update: { drive: { kind: "none" } },
    });
    expect(studioPopulationEdit(population(), "drive.kind", "constant")).toEqual({
      ok: true,
      update: { drive: { current: 0, kind: "constant" } },
    });
    expect(studioPopulationEdit(population(), "drive.kind", "poisson")).toEqual({
      ok: true,
      update: { drive: { kind: "poisson", rate_hz: 0, weight: 0 } },
    });
  });

  it("refuses a drive kind the specification does not admit", () => {
    expect(studioPopulationEdit(population(), "drive.kind", "sinusoid")).toEqual({
      ok: false,
      reason: "Input must be one of none, constant, poisson.",
    });
  });

  it("changes one drive field and leaves the rest of the drive alone", () => {
    const driven = population({ drive: { kind: "poisson", rate_hz: 20, weight: 0.5 } });

    expect(studioPopulationEdit(driven, "drive.rate_hz", "30")).toEqual({
      ok: true,
      update: { drive: { kind: "poisson", rate_hz: 30, weight: 0.5 } },
    });
  });

  it("reads an empty Poisson seed as a request to derive one", () => {
    const driven = population({ drive: { kind: "poisson", rate_hz: 20, seed: 7, weight: 0.5 } });

    expect(studioPopulationEdit(driven, "drive.seed", "")).toEqual({
      ok: true,
      update: { drive: { kind: "poisson", rate_hz: 20, weight: 0.5 } },
    });
  });

  it("changes one parameter and leaves the other overrides alone", () => {
    const overridden = population({ params: { capacitance: 2, tau: 10 } });

    expect(studioPopulationEdit(overridden, "params.tau", "12")).toEqual({
      ok: true,
      update: { params: { capacitance: 2, tau: 12 } },
    });
  });

  it("refuses a field that is not editable rather than writing it", () => {
    expect(studioPopulationEdit(population(), "position", "0")).toEqual({
      ok: false,
      reason: "position is not an editable field.",
    });
    expect(studioPopulationEdit(population(), "position", "left")).toEqual({
      ok: false,
      reason: "position must be a number.",
    });
  });
});
