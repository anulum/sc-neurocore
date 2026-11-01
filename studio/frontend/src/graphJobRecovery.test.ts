// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Actual browser graph recovery and retained input seals
// @vitest-environment happy-dom

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { beforeEach, expect, it } from "vitest";
import {
  graphWorkspaceId, initialGraphWorkspaceId, loadGraphJobRecovery, prepareGraphJobRecovery,
  readGraphJobRecovery, storeGraphJobRecovery, type GraphJobRecovery,
} from "./graphJobRecovery";
import { browserSavedSessionStorage, readStudioSessionsResult } from "./studioSavedSessions";

/** Read the real producer's request/validation pair into retained recovery metadata. */
function fixture(): GraphJobRecovery {
  const value: unknown = JSON.parse(readFileSync(resolve(import.meta.dirname, "test-data/graphJobContract.json"), "utf8"));
  if (value === null || typeof value !== "object" || !("validation" in value) || !("request" in value)) throw new Error("Missing corpus");
  return { schema_version: "studio.graph.job-recovery.v1", owner: "alice", selectedJobId: null,
    submission: { requestJson: JSON.stringify(value.request), retryKey: "original-request", validation: value.validation } };
}

beforeEach(() => { localStorage.clear(); sessionStorage.clear(); });

it("persists exact input/key and restores its independently checked actual server interpretation", async () => {
  const workspace = graphWorkspaceId();
  const recovery = fixture();
  expect(browserSavedSessionStorage()).toBe(localStorage);
  expect(readStudioSessionsResult()).toEqual({ sessions: [], status: "empty", discarded: 0 });
  storeGraphJobRecovery(workspace, recovery);
  expect(readStudioSessionsResult()).toEqual({ sessions: [], status: "empty", discarded: 0 });
  expect(initialGraphWorkspaceId()).toBe(workspace);
  const restored = loadGraphJobRecovery(workspace, "alice");
  expect(restored).toEqual(recovery);
  expect(loadGraphJobRecovery(workspace, "bob")).toBeNull();
  expect(loadGraphJobRecovery(graphWorkspaceId(), "alice")).toBeNull();
  if (restored?.submission === null || restored?.submission === undefined) throw new Error("Missing retained input");
  const prepared = await prepareGraphJobRecovery(restored.submission);
  expect(prepared.request.json).toBe(recovery.submission?.requestJson);
  expect(prepared.identity.projected_neuron_steps).toBe(800);
  expect(prepared.request.inputKey).not.toContain('"position"');
  storeGraphJobRecovery(workspace, { ...recovery, selectedJobId: "sj_0123456789abcdef", submission: null });
  expect(loadGraphJobRecovery(workspace, "alice")).toMatchObject({ selectedJobId: "sj_0123456789abcdef", submission: null });
});

it("assigns fresh UUIDs only to absent/initial invalid context, never to explicitly malformed saved identities", () => {
  expect(initialGraphWorkspaceId()).toMatch(/^[0-9a-f-]{36}$/);
  sessionStorage.setItem("sc-neurocore-current-graph-workspace", "corrupt");
  expect(initialGraphWorkspaceId()).toMatch(/^[0-9a-f-]{36}$/);
  expect(() => graphWorkspaceId("corrupt")).toThrow("identity is invalid");
  expect(() => graphWorkspaceId(null)).toThrow("identity is invalid");
  expect(readGraphJobRecovery(undefined)).toBeNull();
  expect(readGraphJobRecovery(null)).toBeNull();
});

it.each([
  ["schema_version", "unknown"], ["owner", ""], ["owner", 1], ["selectedJobId", "../job"],
  ["selectedJobId", 4], ["submission", 3], ["submission", []],
] as const)("refuses malformed recovery metadata %s", (field, value) => {
  expect(() => readGraphJobRecovery({ ...fixture(), [field]: value })).toThrow("invalid");
});

it.each([
  ["requestJson", ""], ["requestJson", 4], ["requestJson", "x".repeat(16 * 1024 * 1024 + 1)],
  ["retryKey", "space key"], ["retryKey", 3],
] as const)("refuses malformed retained submission field %s", (field, value) => {
  const recovery = fixture();
  expect(() => readGraphJobRecovery({ ...recovery, submission: { ...recovery.submission, [field]: value } })).toThrow("retry input is invalid");
});

it("refuses absent required fields and a foreign actor's actual stored metadata", () => {
  for (const field of ["schema_version", "owner", "selectedJobId", "submission"] as const) {
    const value = { ...fixture() }; Reflect.deleteProperty(value, field);
    expect(() => readGraphJobRecovery(value)).toThrow("invalid");
  }
  const value = fixture();
  if (value.submission === null) throw new Error("Missing input");
  for (const field of ["requestJson", "retryKey", "validation"] as const) {
    const submission = { ...value.submission }; Reflect.deleteProperty(submission, field);
    expect(() => readGraphJobRecovery({ ...value, submission })).toThrow("invalid");
  }
  for (const invalid of [[], 3, "record"]) expect(() => readGraphJobRecovery(invalid)).toThrow("invalid");
  const workspace = graphWorkspaceId();
  storeGraphJobRecovery(workspace, value);
  const key = localStorage.key(0);
  if (key === null) throw new Error("Recovery was not stored");
  localStorage.setItem(key, JSON.stringify({ ...value, owner: "bob" }));
  expect(() => loadGraphJobRecovery(workspace, "alice")).toThrow("another account");
  localStorage.setItem(key, "{broken");
  expect(() => loadGraphJobRecovery(workspace, "alice")).toThrow();
});

it("does not trust a stored validation after input corruption or a different scientific request", async () => {
  const recovery = fixture();
  if (recovery.submission === null) throw new Error("Missing input");
  await expect(prepareGraphJobRecovery({ ...recovery.submission, requestJson: "{broken" })).rejects.toThrow();
  await expect(prepareGraphJobRecovery({ ...recovery.submission, requestJson: '{"populations":[],"projections":[],"seed":43}' })).rejects.toThrow("does not match");
  await expect(prepareGraphJobRecovery({ ...recovery.submission, validation: null })).rejects.toThrow("Invalid asynchronous");
});
