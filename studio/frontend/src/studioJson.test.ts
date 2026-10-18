// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Exact seed JSON across real response and workspace codecs

import { describe, expect, it } from "vitest";
import { json } from "./api/http";
import { parseStudioJson, stringifyStudioJson } from "./studioJson";
import { studioInitialData } from "./stores/studioInitialState";
import {
  studioProjectStateFromLoadResponse, studioProjectRestoreState, studioProjectSaveState,
} from "./studioProjectState";

describe("Studio seed JSON", () => {
  it.each(["9007199254740993", "9223372036854775807"])("keeps numeric seed token %s across the real response/load/restore/save boundary", async (token) => {
    const body = await json<unknown>(new Response(`{"state":{"seed":${token}}}`));
    const loaded = studioProjectStateFromLoadResponse(body, studioInitialData.trainingConfig);
    expect(loaded.seed).toBe(BigInt(token));
    const restored = studioProjectRestoreState(loaded);
    const saved = studioProjectSaveState(restored);
    expect(saved.seed).toBe(BigInt(token));
    const encoded = stringifyStudioJson({ state: saved });
    expect(encoded).toContain(`"seed":${token}`);
    expect(parseStudioJson(encoded)).toMatchObject({ state: { seed: BigInt(token) } });
  });

  it("writes an already represented integral Number with its exact integer digits", () => {
    expect(stringifyStudioJson({ seed: 2 ** 60 })).toBe('{"seed":1152921504606846976}');
    expect(parseStudioJson(stringifyStudioJson({ seed: 2 ** 60 }))).toEqual({ seed: 1152921504606846976n });
  });

  it("preserves safe and legacy JSON, quoted seed text, nested seeds and ordinary decimals", () => {
    const value = { seed: 42, noise: 1.5, text: "9223372036854775807", nested: { seed: 9223372036854775807n } };
    const encoded = stringifyStudioJson(value, 2);
    expect(encoded).toContain('"seed": 9223372036854775807');
    expect(parseStudioJson(encoded)).toEqual(value);
    expect(parseStudioJson('{"seed":"42"}')).toEqual({ seed: "42" });
    expect(parseStudioJson('{"seed":1.5}')).toEqual({ seed: 1.5 });
    expect(parseStudioJson('{"odeParams":{"seed":1.0},"odeInit":{"seed":1e2}}')).toEqual({
      odeParams: { seed: 1 }, odeInit: { seed: 100 },
    });
    expect(parseStudioJson('{"other":1.5,"seed":null}')).toEqual({ other: 1.5, seed: null });
  });

  it("refuses an ambiguous rounded numeric seed token or a non-JSON root", () => {
    expect(() => parseStudioJson('{"seed":9007199254740993.0}')).toThrow("exact integer token");
    expect(() => stringifyStudioJson(undefined)).toThrow("serialisable root");
    expect(() => stringifyStudioJson({ seed: Number.NaN })).toThrow("non-finite seed");
    expect(() => stringifyStudioJson({ seed: Number.POSITIVE_INFINITY })).toThrow("non-finite seed");
    expect(() => parseStudioJson("{")).toThrow();
  });
});

it("refuses large seeds on a runtime without native JSON source access", () => {
  const descriptor = Object.getOwnPropertyDescriptor(JSON, "parse");
  if (!descriptor) throw new Error("The runtime has no JSON parser.");
  const original = JSON.parse.bind(JSON);
  let failure: unknown;
  try {
    Reflect.set(JSON, "parse", (text: string, reviver?: (key: string, value: unknown) => unknown): unknown =>
      original(text, function (this: object, key: string, value: unknown): unknown { return reviver ? reviver.call(this, key, value) : value; }) as unknown);
    try { parseStudioJson('{"seed":9007199254740993}'); }
    catch (error) { failure = error; }
  } finally {
    Object.defineProperty(JSON, "parse", descriptor);
  }
  expect(failure).toBeInstanceOf(Error);
  expect(failure instanceof Error ? failure.message : "").toContain("cannot read large Studio seeds");
});

it.each(["absent", "undefined"])("refuses lossy integer output when raw JSON support is %s", (mode) => {
  const descriptor = Object.getOwnPropertyDescriptor(JSON, "rawJSON");
  if (!descriptor) throw new Error("The test runtime lacks raw JSON support.");
  try {
    if (mode === "absent") Reflect.deleteProperty(JSON, "rawJSON");
    else Reflect.set(JSON, "rawJSON", undefined);
    expect(stringifyStudioJson({ seed: 42 })).toBe('{"seed":42}');
    expect(() => stringifyStudioJson({ seed: 9223372036854775807n })).toThrow("cannot write large Studio seeds");
  } finally {
    Object.defineProperty(JSON, "rawJSON", descriptor);
  }
});

it("keeps protocol seeds exact while projecting ODE coefficient maps to their existing Float64 contract", async () => {
  const wire = await json<unknown>(new Response('{"state":{"sourceMode":"ode","seed":9223372036854775807,"odeParams":{"seed":9007199254740993},"odeInit":{"seed":9007199254740993}}}'));
  const state = studioProjectStateFromLoadResponse(wire, studioInitialData.trainingConfig);
  expect(state.seed).toBe(9223372036854775807n);
  expect(state.odeParams.seed).toBe(Number(9007199254740993n));
  expect(state.odeInit.seed).toBe(Number(9007199254740993n));
});

it("preserves Float64 fields named seed throughout actual ODE request, experiment and observation shapes", () => {
  const value = parseStudioJson('{"seed":9223372036854775807,"odeParams":{"seed":1e20},"odeInit":{"seed":9007199254740993.0},"request":{"equations":["dseed/dt = 0"],"params":{"seed":1e20}},"experiment":{"source":"ode","parameters":{"seed":1e20}},"initial_state":{"seed":1e20},"final_state":{"seed":1e20},"state_ranges":{"seed":[1e20,1e20]},"states":{"seed":[1e20]},"frames":[{"seed":9007199254740993}]}');
  expect(value).toEqual({ seed: 9223372036854775807n, odeParams: { seed: 1e20 },
    odeInit: { seed: Number(9007199254740993n) },
    request: { equations: ["dseed/dt = 0"], params: { seed: 1e20 } },
    experiment: { source: "ode", parameters: { seed: 1e20 } },
    initial_state: { seed: 1e20 }, final_state: { seed: 1e20 }, state_ranges: { seed: [1e20, 1e20] },
    states: { seed: [1e20] }, frames: [{ seed: 9007199254740993n }] });
  expect(parseStudioJson('{"seed":1.5}')).toEqual({ seed: 1.5 });
  expect(parseStudioJson("null")).toBeNull();
  expect(parseStudioJson("42")).toBe(42);
});

it("retains model seed fields while refusing a numeric overflow masquerading as an integral seed", () => {
  expect(parseStudioJson('{"model":{"source":"model","parameters":{"seed":9007199254740993}},"other":{"params":{"seed":9223372036854775807}}}'))
    .toEqual({ model: { source: "model", parameters: { seed: 9007199254740993n } },
      other: { params: { seed: 9223372036854775807n } } });
  expect(() => parseStudioJson('{"seed":1e400}')).toThrow("exact integer token");
});
