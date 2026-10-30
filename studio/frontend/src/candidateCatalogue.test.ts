// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Actual catalogue draft transport identity and text conservation

import { afterAll, beforeAll, expect, it } from "vitest";
import { parseCatalogueCandidate } from "./candidateCatalogue";
import { startCandidateTestServer, type CandidateTestServer } from "./candidateTestServer";

let server: CandidateTestServer;
let response: Record<string, unknown>;
beforeAll(async () => {
  server = await startCandidateTestServer();
  const actual = await fetch(`${server.origin}/api/candidates/from-catalogue/AdExNeuron`,
    { headers: { "sec-fetch-site": "none" } });
  expect(actual.status).toBe(200);
  response = await actual.json() as Record<string, unknown>;
}, 40_000);
afterAll(async () => { await server.stop(); });

it("retains the actual canonical response without rewriting candidate JSON", () => {
  const result = parseCatalogueCandidate(response, "AdExNeuron");
  expect(result.parent_model).toBe("AdExNeuron");
  expect(result.candidate_text).toBe(response.candidate_text);
  expect(result.parent_schema_sha256).toBe(response.parent_schema_sha256);
});

it.each([null, false, "response", 0])("refuses a nonobject catalogue response %j", (value) => {
  expect(() => parseCatalogueCandidate(value, "AdExNeuron")).toThrow("catalogue draft response is invalid");
});

it.each([
  ["schema_version", "unknown"], ["parent_model", "PoissonNeuron"],
  ["parent_schema", 4], ["parent_schema", ""],
  ["parent_schema_sha256", 4], ["parent_schema_sha256", "bad-digest"],
  ["candidate_text", false], ["candidate_text", ""],
])("refuses a changed transport field %s=%j", (key, value) => {
  expect(() => parseCatalogueCandidate({ ...response, [key]: value }, "AdExNeuron"))
    .toThrow("catalogue draft response is invalid");
});
