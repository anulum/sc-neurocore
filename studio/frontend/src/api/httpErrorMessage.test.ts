// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — tests for the sentences read out of server error details

import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { createServer } from "node:http";

import { errorMessage, get, patch, post } from "./http";

describe("server error sentences", () => {
  it("says a numerical model failure in full instead of its code", () => {
    expect(errorMessage({
      error: "model_simulation_failed",
      model: "ATypeKNeuron",
      backend: "python",
      step: 170,
      time_ms: 85,
      diagnostic: "ValueError: membrane candidate left physiological safety bounds",
    }, 422)).toBe(
      "ATypeKNeuron (python) simulation failed at step 170 (85.0 ms): "
        + "ValueError: membrane candidate left physiological safety bounds.",
    );
  });

  it("keeps what it can when fields are missing", () => {
    expect(errorMessage({ error: "model_simulation_failed" }, 422)).toBe("The model simulation failed.");
  });

  it("prefers a reason or message and falls back to the status", () => {
    expect(errorMessage({ reason: "unsupported protocol" }, 422)).toBe("unsupported protocol");
    expect(errorMessage("plain detail", 400)).toBe("plain detail");
    expect(errorMessage(null, 503)).toBe("503");
  });
});

describe("actual HTTP seed transport", () => {
  const requests: { method: string | undefined; body: string }[] = [];
  const nativeFetch = globalThis.fetch.bind(globalThis);
  const server = createServer((request, response) => {
    let body = "";
    request.setEncoding("utf8");
    request.on("data", (part: string) => { body += part; });
    request.on("end", () => {
      requests.push({ method: request.method, body });
      if (request.url === "/api/proxy-error") {
        response.writeHead(503, "Service Unavailable");
        response.end("<html>proxy unavailable</html>");
      } else if (request.url === "/api/refused") {
        response.writeHead(422, { "Content-Type": "application/json" });
        response.end('{"detail":{"reason":"model seed refused","seed":9223372036854775807}}');
      } else {
        response.writeHead(200, { "Content-Type": "application/json" });
        response.end(body || '{"seed":9223372036854775807}');
      }
    });
  });
  beforeAll(async () => {
    await new Promise<void>((resolve) => { server.listen(0, "127.0.0.1", resolve); });
    const address = server.address();
    if (typeof address !== "object" || address === null) throw new Error("HTTP fixture has no address.");
    const origin = `http://127.0.0.1:${address.port}`;
    // Only supply the browser's URL origin; requests and responses use the real network.
    vi.stubGlobal("fetch", (input: string, init?: RequestInit) => nativeFetch(new URL(input, origin), init));
  });
  afterAll(async () => {
    vi.unstubAllGlobals();
    await new Promise<void>((resolve, reject) => { server.close((error) => { if (error) reject(error); else resolve(); }); });
  });
  it("writes POST and PATCH as numeric integer tokens and reads original GET bytes", async () => {
    const seed = 9223372036854775807n;
    expect(await post("/echo", { seed })).toEqual({ seed });
    expect(await patch("/echo", { seed: 9007199254740993n })).toEqual({ seed: 9007199254740993n });
    expect(requests.slice(-2)).toEqual([
      { method: "POST", body: '{"seed":9223372036854775807}' },
      { method: "PATCH", body: '{"seed":9007199254740993}' },
    ]);
    expect(await get("/seed")).toEqual({ seed });
  });
  it("keeps structured refusal details exact and preserves a proxy error fallback", async () => {
    await expect(get("/refused")).rejects.toMatchObject({
      message: "model seed refused", status: 422,
      detail: { reason: "model seed refused", seed: 9223372036854775807n },
    });
    await expect(get("/proxy-error")).rejects.toMatchObject({ message: "Service Unavailable", status: 503 });
  });
});
