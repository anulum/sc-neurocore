// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio synthesis terminal API request

import { afterEach, describe, expect, it, vi } from "vitest";

import type {
  CompileTraceability,
  ModelCompileResultConfiguration,
  ModelCompileSourcePayload,
  ModelCosimReport,
} from "./types";
import { runSynthesisTerminal } from "./synthApi";

describe("runSynthesisTerminal", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("posts the selected RTL and both evidence objects without a host path", async () => {
    const response = { schema_version: "studio.silicon-terminal.v1" };
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(response), {
      headers: { "Content-Type": "application/json" },
    }));
    vi.stubGlobal("fetch", fetchMock);
    const compileTraceability = {
      schema_version: "studio.compile-traceability.v1",
    } as CompileTraceability;
    const cosimParity = {
      schema_version: "studio.cosim-parity.v1",
    } as ModelCosimReport;

    const result = await runSynthesisTerminal(
      "module selected; endmodule",
      "ecp5",
      compileTraceability,
      cosimParity,
    );

    expect(result).toEqual(response);
    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, request] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/synth/terminal");
    expect(request.method).toBe("POST");
    // `body` is `BodyInit`, which stringifies to "[object Object]" for the
    // shapes that are not text. Checking the type states what the transport is
    // supposed to send instead of asserting on that string.
    const body = request.body;
    if (typeof body !== "string") {
      throw new TypeError(`the transport sent a ${typeof body} body, not JSON text`);
    }
    expect(JSON.parse(body)).toEqual({
      compile_traceability: compileTraceability,
      cosim_parity: cosimParity,
      target: "ecp5",
      verilog: "module selected; endmodule",
    });
    expect(body).not.toContain("json_path");
  });

  it("sends a stochastic model's LFSR seed in both evidence objects, unchanged", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({}), {
      headers: { "Content-Type": "application/json" },
    }));
    vi.stubGlobal("fetch", fetchMock);
    // Typed literals, not casts: the compiler refuses a property the types do not name,
    // so the seed the backend records in both objects is part of each type.
    const sourcePayload: ModelCompileSourcePayload = {
      dt: 0.1,
      integrator: "euler",
      model_name: "EscapeRateNeuron",
      params: {},
      q_format: "Q16.16",
      rng_seed: 1234,
      schema_name: "escape_rate",
      schema_sha256: "a".repeat(64),
    };
    const configuration: ModelCompileResultConfiguration = {
      dt: 0.1,
      integrator: "euler",
      model_name: "EscapeRateNeuron",
      q_format: "Q16.16",
      rng_seed: 1234,
      schema_name: "escape_rate",
      schema_sha256: "a".repeat(64),
    };
    const compileTraceability = {
      schema_version: "studio.compile-traceability.v1",
      source_payload: sourcePayload,
    } as CompileTraceability;
    const cosimParity = {
      schema_version: "studio.cosim-parity.v1",
      configuration,
    } as ModelCosimReport;

    await runSynthesisTerminal("module seeded; endmodule", "ice40", compileTraceability, cosimParity);

    const [, request] = fetchMock.mock.calls[0] as [string, RequestInit];
    const body = request.body;
    if (typeof body !== "string") {
      throw new TypeError(`the transport sent a ${typeof body} body, not JSON text`);
    }
    const sent = JSON.parse(body) as {
      compile_traceability: { source_payload: ModelCompileSourcePayload };
      cosim_parity: { configuration: ModelCompileResultConfiguration };
    };
    expect(sent.compile_traceability.source_payload.rng_seed).toBe(1234);
    expect(sent.cosim_parity.configuration.rng_seed).toBe(1234);
  });
});
