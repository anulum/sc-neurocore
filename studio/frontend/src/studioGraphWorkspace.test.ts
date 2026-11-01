// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Saved graph workspace input and restoration boundaries

import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
import { graphWorkspaceId } from "./graphJobRecovery";
import { readGraphJobSavedInput } from "./graphJobSavedInput";
import { restoreStudioGraphWorkspace, saveStudioGraphWorkspace } from "./studioGraphWorkspace";

/** Read the actual native graph request as current editor state. */
function input() {
  const value: unknown = JSON.parse(readFileSync(new URL("./test-data/graphJobContract.json", import.meta.url), "utf8"));
  if (value === null || typeof value !== "object" || !("request" in value)) throw new Error("Missing native input");
  const graph = readGraphJobSavedInput(value.request);
  return { graphPopulations: graph.populations, graphProjections: graph.projections, graphWorkspaceId: graphWorkspaceId(),
    graphJobRecovery: { schema_version: "studio.graph.job-recovery.v1" as const, owner: "alice", selectedJobId: "sj_0123456789abcdef", submission: null } };
}

it("round-trips real editable graph fields and selected job without serialising scientific results", () => {
  const original = input();
  const saved = saveStudioGraphWorkspace(original);
  expect(saved).toEqual(original);
  expect(saved.graphPopulations).not.toBe(original.graphPopulations);
  expect(saved.graphProjections).not.toBe(original.graphProjections);
  const state: unknown = JSON.parse(JSON.stringify({ ...saved, graphSimResult: { success: true }, graphContextToken: "old" }));
  if (state === null || typeof state !== "object" || Array.isArray(state)) throw new Error("Missing state");
  const first = restoreStudioGraphWorkspace(state as Record<string, unknown>);
  const second = restoreStudioGraphWorkspace(state as Record<string, unknown>);
  expect(first).toMatchObject(original);
  expect(first.graphContextToken).not.toBe(second.graphContextToken);
  expect(first.graphSimResult).toBeNull();
  expect(first.graphHistory).toEqual({ past: [], future: [] });
  expect(first.selectedPopulationIds).toEqual([]);
  expect(first.selectedPopulationId).toBeNull();
  expect(first.selectedProjectionId).toBeNull();
  expect(saved).not.toHaveProperty("graphContextToken");
  expect(saved).not.toHaveProperty("graphSimResult");
});

it("opens graph-less server projects as distinct empty workspaces", () => {
  expect(saveStudioGraphWorkspace({})).toEqual({});
  const first = restoreStudioGraphWorkspace({});
  const second = restoreStudioGraphWorkspace({});
  expect(first.graphWorkspaceId).not.toBe(second.graphWorkspaceId);
  expect(first.graphPopulations).toEqual([]);
  expect(first.graphProjections).toEqual([]);
  expect(first.graphJobRecovery).toBeNull();
  expect(saveStudioGraphWorkspace({ graphProjections: [] }).graphPopulations).toEqual([]);
  expect(saveStudioGraphWorkspace({ graphPopulations: [] }).graphProjections).toEqual([]);
});

it.each([{ graphPopulations: null }, { graphProjections: null }, { graphPopulations: [{ id: "broken" }] },
  { graphWorkspaceId: "../other" }, { graphJobRecovery: { owner: "alice" } }])(
  "refuses explicitly unreadable graph state before any restore patch is applied: %j", (patch) => {
    expect(() => restoreStudioGraphWorkspace({ ...input(), ...patch })).toThrow();
  },
);
