// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio synthesis terminal API request

import { afterEach, describe, expect, it, vi } from "vitest";

import type { CompileTraceability, ModelCosimReport } from "./types";
import { runSynthesisTerminal } from "./synthApi";

describe("runSynthesisTerminal", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("posts the selected RTL and both evidence objects without a host path", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      json: async () => ({ schema_version: "studio.silicon-terminal.v1" }),
      ok: true,
    });
    vi.stubGlobal("fetch", fetchMock);
    const compileTraceability = {
      schema_version: "studio.compile-traceability.v1",
    } as CompileTraceability;
    const cosimParity = {
      schema_version: "studio.cosim-parity.v1",
    } as ModelCosimReport;

    await runSynthesisTerminal(
      "module selected; endmodule",
      "ecp5",
      compileTraceability,
      cosimParity,
    );

    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, request] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/synth/terminal");
    expect(request.method).toBe("POST");
    expect(JSON.parse(String(request.body))).toEqual({
      compile_traceability: compileTraceability,
      cosim_parity: cosimParity,
      target: "ecp5",
      verilog: "module selected; endmodule",
    });
    expect(String(request.body)).not.toContain("json_path");
  });
});
