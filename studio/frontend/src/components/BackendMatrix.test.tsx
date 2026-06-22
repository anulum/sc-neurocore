import { describe, expect, it } from "vitest";

import type { ModelBackendSupport } from "../api/client";
import { orderedBackends } from "./BackendMatrix";

const b = (name: string, status: string, parity: string): ModelBackendSupport =>
  ({ name, status, parity }) as ModelBackendSupport;

describe("orderedBackends", () => {
  it("keeps only implemented backends, Python reference first", () => {
    const result = orderedBackends([
      b("mojo", "implemented", "ulp-bounded"),
      b("rust", "planned", "exact"),
      b("python", "implemented", "exact"),
      b("julia", "implemented", "exact"),
    ]);
    expect(result.map((x) => x.name)).toEqual(["python", "julia", "mojo"]);
  });

  it("returns a single backend for Python-only models", () => {
    const result = orderedBackends([b("python", "implemented", "exact")]);
    expect(result.map((x) => x.name)).toEqual(["python"]);
  });
});
