// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Python OpenAPI and hosted TypeScript contract agreement

import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
import { HOST_API_PROTOCOL, HOST_OPERATIONS } from "./hostApiTypes";

/** Read a generated schema node without silently accepting a missing contract. */
function object(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("Missing OpenAPI object");
  return value as Record<string, unknown>;
}
/** Read exact literal values from the generated discriminator or response field. */
function literals(value: unknown): string[] {
  const schema = object(value);
  if (typeof schema.const === "string") return [schema.const];
  if (Array.isArray(schema.enum) && schema.enum.every((item: unknown): item is string => typeof item === "string")) return schema.enum;
  if (Array.isArray(schema.anyOf)) return schema.anyOf.flatMap(literals);
  throw new Error("Missing OpenAPI literal domain");
}

it("keeps the closed frontend operation set and protocol equal to actual generated Python contracts", () => {
  const document: unknown = JSON.parse(readFileSync(new URL("../../../docs/_generated/studio_openapi.json", import.meta.url), "utf8"));
  const schemas = object(object(object(document).components).schemas);
  const names = ["HostAnalysisRequest", "HostIRBuildRequest", "HostIRTextRequest", "HostIREmitRequest"];
  const admitted = names.flatMap((name) => {
    const schema = object(schemas[name]);
    expect(schema.additionalProperties).toBe(false);
    expect(new Set(schema.required as unknown[])).toEqual(new Set(["operation", "request"]));
    return literals(object(schema.properties).operation);
  });
  expect(new Set(admitted)).toEqual(new Set(HOST_OPERATIONS));
  expect(new Set(HOST_OPERATIONS).size).toBe(HOST_OPERATIONS.length);
  const description = object(object(schemas.HostBridgeDescription).properties);
  expect(literals(description.protocol)).toEqual([HOST_API_PROTOCOL]);
  expect(literals(description.profile)).toEqual(["local-first"]);
  expect(new Set(literals(object(description.operations).items))).toEqual(new Set(HOST_OPERATIONS));
});
