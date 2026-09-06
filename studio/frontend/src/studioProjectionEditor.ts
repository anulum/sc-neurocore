// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Editing the fields a projection is actually run with

/**
 * The connection contract, as fields a user can reach.
 *
 * `studio.network-graph-spec.v1` executes a projection's weight, rule,
 * probability, delay, seed and autapse setting. The canvas created every
 * projection from the request-builder defaults and then offered no way to
 * change any of them: the graph ran fields the editor could not edit.
 *
 * This is the model behind those inputs. It does two things and deliberately
 * not a third:
 *
 * * it states each field's contract in the words the server would use, so a
 *   user reads the constraint before breaking it rather than after;
 * * it parses a typed-in value into an update, refusing only what is not a
 *   value at all — a blank box, text where a number belongs;
 * * it does **not** decide whether a value is admissible. A weight's sign
 *   against its source population, a delay against the graph timestep, a
 *   probability's range: those belong to the server, which owns the contract
 *   and reports every failure at once with the field it came from. Re-deciding
 *   them here would be a second implementation of the contract, free to drift
 *   from the one that runs.
 */

import type { PopulationNode, ProjectionEdge, ProjectionRule } from "./api/client";
import type { StudioGraphIssueLocation } from "./studioGraphValidation";

/** The fields of a projection that the runtime executes. */
export const STUDIO_PROJECTION_FIELDS = [
  "weight",
  "rule",
  "probability",
  "delay",
  "seed",
  "autapses",
] as const;

export type StudioProjectionField = (typeof STUDIO_PROJECTION_FIELDS)[number];

/** One editable field: what it is called, what it holds, what it must satisfy. */
export interface StudioProjectionFieldModel {
  field: StudioProjectionField;
  label: string;
  /** `number`, `text` or `checkbox`; `rule` is a choice. */
  kind: "number" | "checkbox" | "choice";
  /** The current value as the input carries it. */
  value: string;
  /** Whether the box is ticked; only meaningful for a checkbox. */
  checked: boolean;
  /** The contract, in the words the server would use to refuse it. */
  help: string;
  /** Choices, for a field that has them. */
  choices: readonly string[];
  /** Whether this field applies to the projection as it currently stands. */
  applies: boolean;
  /** What validation said about this field, if anything. */
  errors: string[];
}

/** A parsed edit, or the reason the text is not a value at all. */
export type StudioProjectionEdit =
  | { readonly ok: true; readonly update: Partial<ProjectionEdge> }
  | { readonly ok: false; readonly reason: string };

/** Rules the graph spec admits, in the order the editor offers them. */
export const STUDIO_PROJECTION_RULES: readonly ProjectionRule[] = ["random", "all_to_all"];

function signHelp(source: PopulationNode | undefined): string {
  if (source === undefined) {
    return "Signed synaptic weight; the sign must agree with the source population's type.";
  }
  return source.neuron_type === "inhibitory"
    ? "Signed synaptic weight; an inhibitory source needs a negative weight."
    : "Signed synaptic weight; an excitatory source needs a positive weight.";
}

function numberText(value: number | undefined): string {
  return value === undefined ? "" : String(value);
}

/**
 * Return the errors reported for one projection, by the field they name.
 *
 * @param issues - Located validation failures for the whole graph.
 * @param projectionId - The projection being edited.
 * @returns Messages per attribute; a failure about the projection as a whole
 *   is filed under the empty attribute, so nothing is lost.
 */
export function studioProjectionFieldErrors(
  issues: readonly StudioGraphIssueLocation[],
  projectionId: string,
): Map<string, string[]> {
  const byField = new Map<string, string[]>();
  for (const issue of issues) {
    if (issue.kind !== "projection" || issue.id !== projectionId) continue;
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
 * Return the editable model of one projection.
 *
 * @param projection - The projection being edited.
 * @param source - Its source population, whose type fixes the weight's sign.
 * @param issues - Located validation failures, so each input can carry its own.
 * @returns One entry per executed field, in a stable order.
 */
export function studioProjectionFields(
  projection: ProjectionEdge,
  source: PopulationNode | undefined,
  issues: readonly StudioGraphIssueLocation[] = [],
): StudioProjectionFieldModel[] {
  const errors = studioProjectionFieldErrors(issues, projection.id);
  const rule = projection.rule ?? "random";
  const of = (field: StudioProjectionField): string[] => errors.get(field) ?? [];
  return [
    {
      applies: true,
      checked: false,
      choices: [],
      errors: of("weight"),
      field: "weight",
      help: signHelp(source),
      kind: "number",
      label: "Weight",
      value: String(projection.weight),
    },
    {
      applies: true,
      checked: false,
      choices: STUDIO_PROJECTION_RULES,
      errors: of("rule"),
      field: "rule",
      help: "random connects each pair with the probability below; all_to_all connects every pair.",
      kind: "choice",
      label: "Rule",
      value: rule,
    },
    {
      applies: rule === "random",
      checked: false,
      choices: [],
      errors: of("probability"),
      field: "probability",
      help: "Connection probability of the random rule, in (0, 1].",
      kind: "number",
      label: "Probability",
      value: numberText(projection.probability),
    },
    {
      applies: true,
      checked: false,
      choices: [],
      errors: of("delay"),
      field: "delay",
      help: "Delay in milliseconds; it must be a whole number of graph timesteps, never rounded.",
      kind: "number",
      label: "Delay (ms)",
      value: String(projection.delay),
    },
    {
      applies: true,
      checked: false,
      choices: [],
      errors: of("seed"),
      field: "seed",
      help: "Connectivity seed; leave it empty to derive one from the graph seed and the edge index.",
      kind: "number",
      label: "Seed",
      value: numberText(projection.seed),
    },
    {
      applies: true,
      checked: projection.autapses === true,
      choices: [],
      errors: of("autapses"),
      field: "autapses",
      help: "A self-projection drops its diagonal unless autapses are declared.",
      kind: "checkbox",
      label: "Autapses",
      value: "",
    },
  ];
}

function parsedNumber(raw: string): number | null {
  const text = raw.trim();
  if (text.length === 0) return null;
  const value = Number(text);
  return Number.isFinite(value) ? value : null;
}

/**
 * Turn a typed-in value into an update to apply, or say why it is not one.
 *
 * Only what is not a value at all is refused here: a blank required box, text
 * where a number belongs. Whether the value is *admissible* is the server's
 * answer, and asking it is what makes the editor honest — a second copy of the
 * contract in the browser is a copy free to drift from the one that runs.
 *
 * @param field - Which field was edited.
 * @param raw - The text or checkbox state the input carries.
 * @returns The update to apply, or the reason the text is not a value.
 */
export function studioProjectionEdit(
  field: StudioProjectionField,
  raw: string | boolean,
): StudioProjectionEdit {
  if (field === "autapses") {
    return { ok: true, update: { autapses: raw === true || raw === "true" } };
  }
  if (field === "rule") {
    const rule = String(raw);
    if (!STUDIO_PROJECTION_RULES.includes(rule as ProjectionRule)) {
      return { ok: false, reason: `Rule must be one of ${STUDIO_PROJECTION_RULES.join(", ")}.` };
    }
    // all_to_all carries no probability: the spec refuses one, so the editor
    // drops it rather than sending a field that would be rejected.
    return rule === "all_to_all"
      ? { ok: true, update: { probability: undefined, rule: "all_to_all" } }
      : { ok: true, update: { rule: "random" } };
  }
  if (field === "seed") {
    if (String(raw).trim().length === 0) {
      // An empty seed is a real choice: derive one from the graph seed.
      return { ok: true, update: { seed: undefined } };
    }
  }
  const value = parsedNumber(String(raw));
  if (value === null) {
    return { ok: false, reason: `${field} must be a number.` };
  }
  return { ok: true, update: { [field]: value } as Partial<ProjectionEdge> };
}

/**
 * Return how the editor names the projection it is editing.
 *
 * The endpoints, because that is how the canvas draws it and how a located
 * validation failure names it.
 */
export function studioProjectionTitle(
  projection: ProjectionEdge,
  populations: readonly PopulationNode[],
): string {
  const labels = new Map(populations.map((population) => [population.id, population.label]));
  const source = labels.get(projection.source) ?? projection.source;
  const target = labels.get(projection.target) ?? projection.target;
  return `${source} → ${target}`;
}
