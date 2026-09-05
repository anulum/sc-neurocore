// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio compiler request state custody tests

import { afterEach, expect, it, vi } from "vitest";
import { useStudioStore } from "./studio";

const initialState = useStudioStore.getState();

afterEach(() => {
  useStudioStore.setState(initialState, true);
  vi.unstubAllGlobals();
});

it.each([
  ["runCompile", "/api/compile"],
  ["runEmitSV", "/api/ir/emit-sv-direct"],
] as const)("%s sends every ODE initial state to %s", async (action, route) => {
  const fetch = vi.fn<typeof globalThis.fetch>().mockResolvedValue(
    new Response(JSON.stringify({ verilog: "module sc_neuron; endmodule", chars: 30 }), {
      headers: { "Content-Type": "application/json" },
    }),
  );
  vi.stubGlobal("fetch", fetch);
  useStudioStore.setState({
    sourceMode: "ode",
    equations: ["dv/dt = -v + w + I", "dw/dt = -w"],
    odeInit: { v: 2, w: 3 },
  });
  await useStudioStore.getState()[action]();
  expect(fetch).toHaveBeenCalledOnce();
  const [url, options] = fetch.mock.calls[0];
  expect(url).toBe(route);
  expect(JSON.parse(String(options?.body))).toMatchObject({
    equations: ["dv/dt = -v + w + I", "dw/dt = -w"],
    init: { v: 2, w: 3 },
  });
});
