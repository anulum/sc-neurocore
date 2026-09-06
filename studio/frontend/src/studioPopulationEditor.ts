// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Editing the fields a population is actually run with

/**
 * The model contract, as fields a user can reach.
 *
 * A population's model, count, type, external drive and constructor parameter
 * overrides are all executed by `studio.network-graph-spec.v1`, and the canvas
 * fixed every one of them at creation. The parameters are the hard half: which
 * constructor fields a population may override, their kind and their default
 * are decided by the run contract, so this reads them from
 * `GET /api/graph/models/{name}` rather than guessing. A browser that guessed
 * would be a second implementation of the contract, free to drift from the one
 * that validates the graph.
 *
 * The same division as the projection editor: this parses, the server decides.
 * A blank box or text where a number belongs is not a value; whether a value
 * is *admissible* is the server's answer, reported with the field it came from.
 */

import type {
  PopulationDrive,
  PopulationModelContract,
  PopulationNode,
  StudioNeuronType,
} from "./api/client";
import type { StudioGraphIssueLocation } from "./studioGraphValidation";

/** The drive kinds the graph specification admits. */
export const STUDIO_DRIVE_KINDS = ["none", "constant", "poisson"] as const;

/** The neuron types a population may declare. */
export const STUDIO_NEURON_TYPES: readonly StudioNeuronType[] = ["excitatory", "inhibitory"];

/** One editable field of a population. */
export interface StudioPopulationFieldModel {
  /** Request attribute this field writes, as the server names it. */
  field: string;
  label: string;
  kind: "text" | "number" | "choice";
  value: string;
  help: string;
  choices: readonly string[];
  errors: string[];
}

/** A parsed edit, or the reason the text is not a value at all. */
export type StudioPopulationEdit =
  | { readonly ok: true; readonly update: Partial<PopulationNode> }
  | { readonly ok: false; readonly reason: string };

/**
 * Return a drive's kind, treating a population saved before drives as undriven.
 *
 * @param drive - The population's drive, absent on a workspace saved before
 *   drives existed.
 * @returns The kind name the graph specification uses.
 */
function driveKind(drive: PopulationDrive | undefined): string {
  return drive?.kind ?? "none";
}

/**
 * Render one numeric field of a drive, leaving an absent one empty.
 *
 * @param drive - The population's drive, if it has one.
 * @param key - The field to read, as the graph specification names it.
 * @returns The value as an input carries it; empty when the drive does not
 *   carry that field, which is how a derived seed is shown.
 */
function driveNumber(drive: PopulationDrive | undefined, key: string): string {
  if (drive === undefined) return "";
  const value = (drive as unknown as Record<string, unknown>)[key];
  return typeof value === "number" ? String(value) : "";
}

/**
 * Return the errors reported for one population, by the attribute they name.
 *
 * A parameter failure arrives as `params.tau`, so it is filed under that whole
 * path; a failure about the population as a whole is filed under the empty
 * attribute, where the editor shows it rather than losing it.
 *
 * @param issues - Located failures for the whole graph.
 * @param populationId - The population being edited.
 * @returns Messages by attribute, in the order the server reported them.
 */
export function studioPopulationFieldErrors(
  issues: readonly StudioGraphIssueLocation[],
  populationId: string,
): Map<string, string[]> {
  const byField = new Map<string, string[]>();
  for (const issue of issues) {
    if (issue.kind !== "population" || issue.id !== populationId) continue;
    const existing = byField.get(issue.attribute);
    if (existing === undefined) {
      byField.set(issue.attribute, [issue.message]);
    } else {
      existing.push(issue.message);
    }
  }
  return byField;
}

/**
 * Return the identity fields of one population: what it is, not what drives it.
 *
 * @param population - The population being edited.
 * @param models - Model names the server admits for a population.
 * @param issues - Located validation failures, so each input carries its own.
 * @returns One entry per identity field, in a stable order.
 */
export function studioPopulationFields(
  population: PopulationNode,
  models: readonly string[],
  issues: readonly StudioGraphIssueLocation[] = [],
): StudioPopulationFieldModel[] {
  const errors = studioPopulationFieldErrors(issues, population.id);
  const of = (field: string): string[] => errors.get(field) ?? [];
  return [
    {
      choices: [],
      errors: of("label"),
      field: "label",
      help: "Display name; it never reaches the run.",
      kind: "text",
      label: "Label",
      value: population.label,
    },
    {
      choices: models,
      errors: of("model"),
      field: "model",
      help: "Catalogue model; only models the graph can execute are offered.",
      kind: "choice",
      label: "Model",
      value: population.model,
    },
    {
      choices: [],
      errors: of("count"),
      field: "count",
      help: "Number of neurons; a positive whole number.",
      kind: "number",
      label: "Neurons",
      value: String(population.count),
    },
    {
      choices: STUDIO_NEURON_TYPES,
      errors: of("neuron_type"),
      field: "neuron_type",
      help: "Fixes the sign every outgoing projection's weight must carry.",
      kind: "choice",
      label: "Type",
      value: population.neuron_type,
    },
  ];
}

/**
 * Return the external-input fields that apply to this population's drive kind.
 *
 * A drive of kind `none` carries no fields at all — the specification refuses
 * one that does — so the editor offers none rather than showing boxes whose
 * values would be rejected.
 *
 * @param population - The population being edited.
 * @param issues - Located validation failures, so each input carries its own.
 * @returns The kind, and the fields that kind carries; nothing else.
 */
export function studioPopulationDriveFields(
  population: PopulationNode,
  issues: readonly StudioGraphIssueLocation[] = [],
): StudioPopulationFieldModel[] {
  const errors = studioPopulationFieldErrors(issues, population.id);
  const of = (field: string): string[] => errors.get(field) ?? [];
  const kind = driveKind(population.drive);
  const fields: StudioPopulationFieldModel[] = [
    {
      choices: STUDIO_DRIVE_KINDS,
      errors: of("drive.kind"),
      field: "drive.kind",
      help: "External input: none, a constant current, or a Poisson spike train.",
      kind: "choice",
      label: "Input",
      value: kind,
    },
  ];
  if (kind === "constant") {
    fields.push({
      choices: [],
      errors: of("drive.current"),
      field: "drive.current",
      help: "Current injected into every neuron at every step; a finite number.",
      kind: "number",
      label: "Current",
      value: driveNumber(population.drive, "current"),
    });
  }
  if (kind === "poisson") {
    fields.push(
      {
        choices: [],
        errors: of("drive.rate_hz"),
        field: "drive.rate_hz",
        help: "Rate of each neuron's independent Poisson process, in hertz.",
        kind: "number",
        label: "Rate (Hz)",
        value: driveNumber(population.drive, "rate_hz"),
      },
      {
        choices: [],
        errors: of("drive.weight"),
        field: "drive.weight",
        help: "Current injected on each Poisson event.",
        kind: "number",
        label: "Event weight",
        value: driveNumber(population.drive, "weight"),
      },
      {
        choices: [],
        errors: of("drive.seed"),
        field: "drive.seed",
        help: "Seed of the Poisson draw; leave it empty to derive one.",
        kind: "number",
        label: "Seed",
        value: driveNumber(population.drive, "seed"),
      },
    );
  }
  return fields;
}

/**
 * Return one input per parameter the model contract says may be overridden.
 *
 * A parameter the population does not override shows the model's declared
 * default, so a user sees what is running rather than an empty box; the value
 * only enters `params` once they change it.
 *
 * @param population - The population being edited.
 * @param contract - The model contract, or `null` while it is being fetched.
 * @param issues - Located validation failures, so each input carries its own.
 * @returns One entry per overridable parameter, in the contract's order; none
 *   while the contract is absent or belongs to another model.
 */
export function studioPopulationParameterFields(
  population: PopulationNode,
  contract: PopulationModelContract | null,
  issues: readonly StudioGraphIssueLocation[] = [],
): StudioPopulationFieldModel[] {
  if (contract?.model !== population.model) {
    return [];
  }
  const errors = studioPopulationFieldErrors(issues, population.id);
  return contract.parameters.map((parameter) => {
    // A record's index type promises a number for every key; the object need
    // not carry one. Asking what is actually there is the difference between
    // "overridden to 0" and "not overridden at all".
    const overridden = Object.prototype.hasOwnProperty.call(
      population.params,
      parameter.name,
    )
      ? population.params[parameter.name]
      : undefined;
    const declared = parameter.default === null ? "" : String(parameter.default);
    return {
      choices: [],
      errors: errors.get(`params.${parameter.name}`) ?? [],
      field: `params.${parameter.name}`,
      help:
        parameter.kind === "int"
          ? `Whole number; the model declares ${declared === "" ? "no default" : declared}.`
          : `The model declares ${declared === "" ? "no default" : declared}.`,
      kind: "number",
      label: parameter.name,
      value: overridden === undefined ? declared : String(overridden),
    };
  });
}

/**
 * Return the fields the contract says are **not** inputs, and why.
 *
 * Showing them is the point: a user who cannot find `profile` should read that
 * it is a non-numeric field rather than conclude the editor is incomplete.
 *
 * @param contract - The model contract, or `null` while it is being fetched.
 * @returns Each field that is not an input, with the reason it is not; empty
 *   while there is no contract to read them from.
 */
export function studioPopulationUnsupported(
  contract: PopulationModelContract | null,
): { name: string; reason: string }[] {
  return contract === null ? [] : contract.unsupported.map((entry) => ({ ...entry }));
}

/**
 * Return the finite number a box holds, or `null` when it holds no value.
 *
 * Blank text and a non-finite result are both `null`: neither is a value the
 * graph specification would carry.
 *
 * @param raw - The text the input carries.
 * @returns The number, or `null` when the text is not one.
 */
function parsedNumber(raw: string): number | null {
  const text = raw.trim();
  if (text.length === 0) return null;
  const value = Number(text);
  return Number.isFinite(value) ? value : null;
}

/**
 * Return the drive with one field changed, keeping the rest of it.
 *
 * A drive is an object of its kind's fields; editing one by replacing the
 * whole drive would silently drop the others.
 *
 * @param population - The population whose drive is being edited.
 * @param key - The drive field to change.
 * @param value - Its new value, or `undefined` to remove the field, which is
 *   how an emptied seed asks for a derived one.
 * @returns The update to apply to the population.
 */
function withDrive(
  population: PopulationNode,
  key: string,
  value: number | undefined,
): Partial<PopulationNode> {
  const current = { ...(population.drive ?? { kind: "none" }) } as Record<string, unknown>;
  const kept =
    value === undefined
      ? Object.fromEntries(Object.entries(current).filter(([name]) => name !== key))
      : { ...current, [key]: value };
  return { drive: kept as unknown as PopulationDrive };
}

/**
 * Turn a typed-in value into an update to apply, or say why it is not one.
 *
 * @param population - The population being edited; a drive or parameter edit
 *   is a change to one key of an object the rest of which must survive.
 * @param field - The request attribute that was edited.
 * @param raw - The text the input carries.
 * @returns The update to apply, or the reason the text is not a value.
 */
export function studioPopulationEdit(
  population: PopulationNode,
  field: string,
  raw: string,
): StudioPopulationEdit {
  if (field === "label") {
    return raw.trim().length === 0
      ? { ok: false, reason: "label must not be empty." }
      : { ok: true, update: { label: raw } };
  }
  if (field === "model") {
    // The parameters of the previous model mean nothing to the new one, and
    // the graph refuses a parameter the model does not declare.
    return { ok: true, update: { model: raw, params: {} } };
  }
  if (field === "neuron_type") {
    return raw === "excitatory" || raw === "inhibitory"
      ? { ok: true, update: { neuron_type: raw } }
      : { ok: false, reason: `neuron_type must be one of ${STUDIO_NEURON_TYPES.join(", ")}.` };
  }
  if (field === "drive.kind") {
    if (!STUDIO_DRIVE_KINDS.includes(raw as (typeof STUDIO_DRIVE_KINDS)[number])) {
      return { ok: false, reason: `Input must be one of ${STUDIO_DRIVE_KINDS.join(", ")}.` };
    }
    // A drive carries only the fields of its own kind: the specification
    // refuses a kind none that carries any, so the others are dropped.
    if (raw === "none") return { ok: true, update: { drive: { kind: "none" } } };
    if (raw === "constant") return { ok: true, update: { drive: { current: 0, kind: "constant" } } };
    return { ok: true, update: { drive: { kind: "poisson", rate_hz: 0, weight: 0 } } };
  }
  const value = parsedNumber(raw);
  if (field === "drive.seed" && raw.trim().length === 0) {
    return { ok: true, update: withDrive(population, "seed", undefined) };
  }
  if (value === null) {
    return { ok: false, reason: `${field} must be a number.` };
  }
  if (field.startsWith("drive.")) {
    return { ok: true, update: withDrive(population, field.slice("drive.".length), value) };
  }
  if (field.startsWith("params.")) {
    return {
      ok: true,
      update: { params: { ...population.params, [field.slice("params.".length)]: value } },
    };
  }
  if (field === "count") {
    return { ok: true, update: { count: value } };
  }
  return { ok: false, reason: `${field} is not an editable field.` };
}
