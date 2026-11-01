// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Saved native graph input and malformed editor fields

import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
import { readGraphJobSavedInput } from "./graphJobSavedInput";

/** Read the actual producer request afresh for each saved-input contract case. */
function fixture(): unknown {
  const sample: unknown = JSON.parse(readFileSync(new URL("./test-data/graphJobContract.json", import.meta.url), "utf8"));
  if (sample === null || typeof sample !== "object" || !("request" in sample)) throw new Error("Missing native request");
  return sample.request;
}

/** Change one original saved JSON field without manufacturing a valid typed graph. */
function change(input: unknown, path: string, value: unknown): unknown {
  let parent = input;
  const keys = path.split(".");
  for (const key of keys.slice(0, -1)) {
    if (parent === null || typeof parent !== "object") throw new Error("Missing container");
    parent = (parent as Record<string, unknown>)[key];
  }
  if (parent === null || typeof parent !== "object") throw new Error("Missing field parent");
  (parent as Record<string, unknown>)[keys.at(-1) ?? ""] = value;
  return input;
}

it("reopens every provided field of the real native request as detached editable input", () => {
  const saved = fixture();
  const graph = readGraphJobSavedInput(saved);
  expect(graph).toEqual(saved);
  const first = graph.populations[0];
  if (first === undefined) throw new Error("Missing population");
  first.params.tau = 11;
  expect(graph).not.toEqual(saved);
});

it("retains legacy omissions and invalid scientific drafts without granting execution", () => {
  const graph = readGraphJobSavedInput(fixture());
  delete graph.duration; delete graph.dt; delete graph.seed;
  for (const population of graph.populations) delete population.drive;
  for (const projection of graph.projections) {
    delete projection.rule; delete projection.probability; delete projection.seed; delete projection.autapses;
  }
  const first = graph.populations[0];
  if (first === undefined) throw new Error("Missing population");
  first.count = -1;
  const restored = readGraphJobSavedInput(graph);
  expect(restored).toEqual(graph);
  expect(restored).not.toHaveProperty("dt");
  expect(restored.populations[0]?.count).toBe(-1);
});

it.each([
  ["populations", null], ["projections", null], ["populations.0", null], ["populations.0.id", 1],
  ["populations.0.type", "other"], ["populations.0.count", "2"], ["populations.0.neuron_type", "unknown"],
  ["populations.0.position", null], ["populations.0.position.x", Infinity], ["populations.0.position.y", "10"],
  ["populations.0.params", []], ["populations.0.params.tau", NaN], ["populations.0.drive", null],
  ["populations.0.drive.kind", "unknown"], ["populations.0.drive.current", "2"],
  ["populations.1.drive.rate_hz", null], ["populations.1.drive.weight", null], ["populations.1.drive.seed", "9"],
  ["projections.0", null], ["projections.0.id", null], ["projections.0.weight", Infinity],
  ["projections.0.delay", "0"], ["projections.0.rule", "unknown"], ["projections.0.probability", null],
  ["projections.0.seed", null], ["projections.0.autapses", "false"],
  ["duration", null], ["dt", null], ["seed", null],
] as const)("refuses unreadable saved field %s without silently replacing it", (path, value) => {
  expect(() => readGraphJobSavedInput(change(fixture(), path, value))).toThrow("unreadable");
});

it.each([null, [], 3, "graph"])("refuses a non-graph document %s", (value) => {
  expect(() => readGraphJobSavedInput(value)).toThrow("unreadable");
});
