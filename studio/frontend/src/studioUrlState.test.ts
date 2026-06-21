// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio share URL state codec tests

import { describe, expect, it } from "vitest";

import {
  buildStudioShareUrl,
  decodeStudioStartupHash,
  encodeStudioSharePayload,
  studioShareUrlPayload,
  type StudioShareUrlInput,
} from "./studioUrlState";

const input: StudioShareUrlInput = {
  sourceMode: "model",
  selectedModelName: "lif",
  equations: ["dv/dt = -v / tau"],
  threshold: "v > 1",
  reset: "v = 0",
  modelParams: { tau: 10 },
  odeParams: { tau: 20 },
  odeInit: { v: 0 },
  dt: 0.1,
  duration: 100,
  current: 10,
  protocol: "constant",
};

describe("Studio share URL state codec", () => {
  it("builds the compact model-mode payload from the active Studio state", () => {
    expect(studioShareUrlPayload(input)).toEqual({
      m: "model",
      mn: "lif",
      eq: ["dv/dt = -v / tau"],
      th: "v > 1",
      rs: "v = 0",
      p: { tau: 10 },
      i: { v: 0 },
      dt: 0.1,
      d: 100,
      c: 10,
      pr: "constant",
    });
  });

  it("uses ODE parameters when sharing an ODE-mode state", () => {
    expect(studioShareUrlPayload({ ...input, sourceMode: "ode" }).p).toEqual({ tau: 20 });
  });

  it("builds an absolute share URL with an injected encoder", () => {
    const url = buildStudioShareUrl(
      input,
      { origin: "https://studio.example", pathname: "/workbench" },
      (payload) => `encoded:${payload}`,
    );

    expect(url.startsWith("https://studio.example/workbench#encoded:")).toBe(true);
    expect(url).toContain("\"mn\":\"lif\"");
  });

  it("decodes startup hash state and normalises optional defaults", () => {
    const payload = { m: "model", mn: "lif", c: 25, d: 250, pr: "burst" };

    expect(decodeStudioStartupHash("#payload", () => JSON.stringify(payload))).toEqual({
      selectedModelName: "lif",
      current: 25,
      duration: 250,
      protocol: "burst",
    });
  });

  it("rejects empty, malformed, or incomplete startup hashes", () => {
    expect(decodeStudioStartupHash("")).toBeNull();
    expect(decodeStudioStartupHash("#bad", () => "{")).toBeNull();
    expect(decodeStudioStartupHash("#bad", () => JSON.stringify({ m: "model" }))).toBeNull();
    expect(decodeStudioStartupHash("#bad", () => JSON.stringify({ m: "bad", mn: "lif" })))
      .toBeNull();
  });

  it("falls back to runtime defaults for zero or invalid numeric hash fields", () => {
    const payload = { m: "model", mn: "lif", c: 0, d: Number.NaN, pr: "" };

    expect(decodeStudioStartupHash("#payload", () => JSON.stringify(payload))).toEqual({
      selectedModelName: "lif",
      current: 10,
      duration: 100,
      protocol: "constant",
    });
  });

  it("keeps payload encoding independent from the browser global", () => {
    expect(encodeStudioSharePayload(studioShareUrlPayload(input), (payload) => payload))
      .toContain("\"m\":\"model\"");
  });
});
