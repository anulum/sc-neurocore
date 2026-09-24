// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Candidate workbench helpers

import { describe, expect, it } from "vitest";

import type { CandidateDiff } from "./api/candidatesApi";
import { StudioRequestError } from "./api/http";
import {
  candidateDiffLines,
  candidateDiffNotice,
  candidateFileName,
  candidateRefusal,
  parseCandidateText,
} from "./candidateWorkbench";

describe("parseCandidateText", () => {
  it("reads a JSON draft", () => {
    expect(parseCandidateText('{"name": "X"}')).toEqual({ ok: true, document: { name: "X" } });
  });

  it("says an empty draft is empty rather than invalid", () => {
    expect(parseCandidateText("  \n")).toEqual({
      ok: false,
      message: "The draft is empty: import a candidate package or write one.",
    });
  });

  it("carries the parser's own message for a draft that is not JSON", () => {
    const parsed = parseCandidateText('{"name": ');
    expect(parsed.ok).toBe(false);
    expect(parsed.ok ? "" : parsed.message).toMatch(/^The draft is not JSON: /);
  });
});

describe("candidateRefusal", () => {
  const validation = { schema_version: "v1", valid: false, candidate_sha256: "a", diagnostics: [] };

  it("reads the validation out of an invalid-candidate refusal", () => {
    const error = new StudioRequestError("invalid_candidate", 422, {
      reason: "invalid_candidate",
      validation,
    });
    expect(candidateRefusal(error)).toEqual(validation);
  });

  it("returns nothing for any other failure", () => {
    expect(candidateRefusal(new Error("network"))).toBeNull();
    expect(candidateRefusal(new StudioRequestError("x", 500, { reason: "invalid_candidate" }))).toBeNull();
    expect(candidateRefusal(new StudioRequestError("x", 422, "text"))).toBeNull();
    expect(candidateRefusal(new StudioRequestError("x", 422, null))).toBeNull();
    expect(candidateRefusal(new StudioRequestError("x", 422, { reason: "other" }))).toBeNull();
  });
});

describe("candidateDiffLines", () => {
  const compared: CandidateDiff = {
    schema_version: "d1",
    parent: "AdExNeuron",
    status: "compared",
    state: [
      { name: "v", status: "unchanged", parent: -65, candidate: -65 },
      { name: "u", status: "added", candidate: 0 },
    ],
    parameters: [
      { name: "tau_w", status: "changed", parent: 100, candidate: 300 },
      { name: "b", status: "removed", parent: 7 },
    ],
    dynamics: [
      { variable: "v", status: "equivalent", parent: "a", candidate: "b" },
      { variable: "w", status: "changed", difference: "w/tau_w" },
      { variable: "u", status: "added", candidate: "-u" },
      { variable: "z", status: "removed", parent: "-z" },
      { variable: "q", status: "not_comparable", reason: "uses IfExp, which has no symbolic reading" },
      { variable: "k", status: "undecided" },
      { variable: "m", status: "unchanged" },
    ],
    reset: [],
    threshold: { status: "not_comparable", reason: "uses Compare, which has no symbolic reading" },
    integration: { status: "changed", changed_fields: ["method", "dt"] },
  };

  it("lists every change and leaves out what is kept", () => {
    expect(candidateDiffLines(compared)).toEqual([
      { section: "state", name: "u", status: "added", detail: "added: 0" },
      { section: "parameters", name: "tau_w", status: "changed", detail: "100 → 300" },
      { section: "parameters", name: "b", status: "removed", detail: "removed (was 7)" },
      { section: "dynamics", name: "v", status: "equivalent", detail: "rewritten, mathematically equal" },
      { section: "dynamics", name: "w", status: "changed", detail: "candidate − parent = w/tau_w" },
      { section: "dynamics", name: "u", status: "added", detail: "added: -u" },
      { section: "dynamics", name: "z", status: "removed", detail: "removed (was -z)" },
      {
        section: "dynamics",
        name: "q",
        status: "not_comparable",
        detail: "uses IfExp, which has no symbolic reading",
      },
      { section: "dynamics", name: "k", status: "undecided", detail: "" },
      {
        section: "threshold",
        name: "condition",
        status: "not_comparable",
        detail: "uses Compare, which has no symbolic reading",
      },
      {
        section: "integration",
        name: "method, dt",
        status: "changed",
        detail: "numerical profile differs from the parent's",
      },
    ]);
  });

  it("states a changed threshold's difference, and nothing for a kept one", () => {
    const changed = candidateDiffLines({
      schema_version: "d1",
      parent: "P",
      status: "compared",
      threshold: { status: "changed", difference: "1" },
    });
    expect(changed).toEqual([
      { section: "threshold", name: "condition", status: "changed", detail: "1" },
    ]);
    const bare = candidateDiffLines({
      schema_version: "d1",
      parent: "P",
      status: "compared",
      threshold: { status: "added" },
      integration: { status: "unchanged", changed_fields: [] },
      dynamics: [{ variable: "v", status: "added" }, { variable: "w", status: "removed" }],
    });
    expect(bare.map((line) => line.detail)).toEqual([
      "added: ",
      "removed (was )",
      "",
    ]);
    expect(candidateDiffLines({ schema_version: "d1", parent: null, status: "no_parent" })).toEqual([]);
  });
});

describe("candidateDiffNotice", () => {
  it("says why there is nothing to diff", () => {
    expect(candidateDiffNotice({ schema_version: "d", parent: null, status: "no_parent" })).toBe(
      "The candidate names no parent, so there is nothing to diff against.",
    );
    expect(
      candidateDiffNotice({ schema_version: "d", parent: "ATypeKNeuron", status: "parent_has_no_schema" }),
    ).toBe("ATypeKNeuron has no canonical schema to compare with.");
    expect(
      candidateDiffNotice({ schema_version: "d", parent: null, status: "parent_has_no_schema" }),
    ).toBe("The parent has no canonical schema to compare with.");
    expect(candidateDiffNotice({ schema_version: "d", parent: "P", status: "compared" })).toBeNull();
  });
});

describe("candidateFileName", () => {
  it("names files after the candidate, or generically when it has no usable name", () => {
    expect(candidateFileName({ name: "SlowAdEx" }, "candidate")).toBe("SlowAdEx.candidate.json");
    expect(candidateFileName({ name: "../etc/passwd" }, "review")).toBe("candidate.review.json");
    expect(candidateFileName(null, "candidate")).toBe("candidate.candidate.json");
    expect(candidateFileName({ name: 7 }, "review")).toBe("candidate.review.json");
  });
});
