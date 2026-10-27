// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — standalone public entry and backend-unavailable workspace
// @vitest-environment happy-dom

import { act, StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { expect, it, vi } from "vitest";
import SnnStudioPanel from "./SnnStudioPanel";
import "./StandaloneStudioPanel";

it("top-level entry retains the real workspace and API failures do not turn it into a hosted view", async () => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("parent", window);
  const requests: string[] = [];
  let restored = false;
  vi.stubGlobal("fetch", vi.fn<typeof fetch>(async (input) => {
    requests.push(typeof input === "string" ? input : input instanceof URL ? input.href : input.url);
    if (restored && requests.at(-1) === "/api/models") return Response.json([]);
    if (restored && requests.at(-1) === "/api/presets") return Response.json([
      { id: "restored-preset", title: "Restored experiment", description: "Transport recovery", suggested_view: "trace" },
    ]);
    return new Response(JSON.stringify({ detail: "Backend unavailable for this transport-failure test." }),
      { status: 503, headers: { "Content-Type": "application/json" } });
  }));
  const post = vi.spyOn(window, "postMessage");
  const host = document.createElement("div"); document.body.append(host);
  const root = createRoot(host);
  try {
    expect(window.parent).toBe(window);
    await act(async () => {
      root.render(<StrictMode><SnnStudioPanel /></StrictMode>);
    });
    expect(host.querySelector('[aria-label="Save project"]')).not.toBeNull();
    expect(host.querySelector('[aria-label="Save session"]')).not.toBeNull();
    expect(requests.some((url) => url.startsWith("/api/"))).toBe(true);
    expect(host.textContent).not.toContain("Hosted view · no backend connected");
    expect(host.textContent).toContain("Model library could not be initialized");
    expect(host.textContent).toContain("Experiment library could not be initialized");
    const apiRetry = host.querySelector<HTMLButtonElement>('[data-testid="api-unreachable"] button');
    expect(apiRetry).not.toBeNull();
    await act(async () => { apiRetry?.click(); });
    expect(host.textContent).toContain("Model library could not be initialized");
    expect(host.textContent).toContain("Experiment library could not be initialized");
    restored = true;
    const modelRetry = host.querySelector<HTMLButtonElement>('[aria-label="Retry model library"]');
    const presetRetry = host.querySelector<HTMLButtonElement>('[aria-label="Retry experiment library"]');
    expect(modelRetry).not.toBeNull();
    expect(presetRetry).not.toBeNull();
    await act(async () => { modelRetry?.click(); presetRetry?.click(); });
    expect(host.textContent).not.toContain("Model library could not be initialized");
    expect(host.textContent).not.toContain("Experiment library could not be initialized");
    expect(host.querySelector('[aria-label="Load experiment Restored experiment"]')).not.toBeNull();
    expect(requests.filter((url) => url === "/api/models").length).toBeGreaterThan(2);
    expect(requests.filter((url) => url === "/api/presets").length).toBeGreaterThan(2);
    expect(post.mock.calls.some((call) => {
      const data: unknown = call[0];
      return typeof data === "object" && data !== null
        && "type" in data && data.type === "scpn:bridge:request";
    })).toBe(false);
  } finally {
    await act(async () => { root.unmount(); });
    document.body.replaceChildren();
    vi.restoreAllMocks(); vi.unstubAllGlobals();
  }
});
