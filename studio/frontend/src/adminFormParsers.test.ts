// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { describe, expect, it } from "vitest";

import {
  boundedInteger,
  evidenceBundleRequestFromForm,
  identityUpdateFromForm,
  jsonObjects,
  optionalText,
  textList,
} from "./adminFormParsers";

describe("textList", () => {
  it("splits, trims, and drops empty tokens", () => {
    expect(textList("a, b , ,c")).toEqual(["a", "b", "c"]);
    expect(textList(null)).toEqual([]);
  });
});

describe("optionalText", () => {
  it("returns null for blank values", () => {
    expect(optionalText("  ")).toBeNull();
    expect(optionalText(null)).toBeNull();
    expect(optionalText("note")).toBe("note");
  });
});

describe("boundedInteger", () => {
  it("clamps truncated finite numbers and falls back on non-finite input", () => {
    expect(boundedInteger("50", 100, 1, 1000)).toBe(50);
    expect(boundedInteger("0", 100, 1, 1000)).toBe(1);
    expect(boundedInteger("9999", 100, 1, 1000)).toBe(1000);
    expect(boundedInteger("nope", 100, 1, 1000)).toBe(100);
    expect(boundedInteger("12.9", 100, 1, 1000)).toBe(12);
  });
});

describe("jsonObjects", () => {
  it("accepts a single object, object arrays, and rejects invalid payloads", () => {
    expect(jsonObjects('{"a":1}')).toEqual([{ a: 1 }]);
    expect(jsonObjects('[{"a":1},2,null,{"b":2}]')).toEqual([{ a: 1 }, { b: 2 }]);
    expect(jsonObjects("not-json")).toEqual([]);
    expect(jsonObjects("42")).toEqual([]);
    expect(jsonObjects("")).toEqual([]);
  });
});

describe("identityUpdateFromForm", () => {
  it("maps active checkbox and roles list", () => {
    const form = new FormData();
    form.set("active", "on");
    form.set("roles", "studio.admin, studio.operator");
    expect(identityUpdateFromForm(form)).toEqual({
      active: true,
      expires_at_utc: null,
      roles: ["studio.admin", "studio.operator"],
    });
  });
});

describe("evidenceBundleRequestFromForm", () => {
  it("builds the create-bundle contract from form fields", () => {
    const form = new FormData();
    form.set("auditLimit", "25");
    form.set("includeAudit", "on");
    form.set("jobIds", "job-a, job-b");
    form.set("projectName", "demo");
    form.set("analysisResults", '[{"kind":"fi"}]');
    form.set("replayMethod", "POST");
    form.set("replayRoute", "/api/x");
    form.set("requestSha256", "a".repeat(64));
    form.set("operatorNote", "replay note");

    const request = evidenceBundleRequestFromForm(form);
    expect(request.audit_limit).toBe(25);
    expect(request.include_audit).toBe(true);
    expect(request.job_ids).toEqual(["job-a", "job-b"]);
    expect(request.project_name).toBe("demo");
    expect(request.analysis_results).toEqual([{ kind: "fi" }]);
    expect(request.command_replay).toEqual({
      method: "POST",
      route: "/api/x",
      request_sha256: "a".repeat(64),
      note: "replay note",
    });
  });
});
