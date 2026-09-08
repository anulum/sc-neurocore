// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Synthesis request currency through public store actions

import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useStudioStore } from "./studio";
const initial = useStudioStore.getState();
const actions = ["runSynthesis", "runMultiTargetSynthesis"] as const;
type Action = typeof actions[number];
beforeEach(() => { useStudioStore.setState({ sourceMode: "ode", verilogSrc: "module current; endmodule" }); });
afterEach(() => { useStudioStore.setState(initial, true); vi.unstubAllGlobals(); });

/**
 * Hold synthesis only, answering operator refresh without a real runner.
 *
 * @param action - Single or multi-target response to provide.
 * @returns A completion handle; fixtures are not measurement evidence.
 */
function transport(action: Action) {
  let finish = (_fail?: boolean): void => { throw new Error("request absent"); };
  const fetch = vi.fn<typeof globalThis.fetch>((_url, init) => {
    if (init?.method !== "POST") return Promise.resolve(new Response(JSON.stringify({ audit: {}, jobs: [] })));
    return new Promise<Response>((resolve, reject) => {
      finish = (fail = false) => {
        if (fail) reject(new Error("runner unavailable"));
        else resolve(new Response(JSON.stringify(action === "runSynthesis"
          ? { success: true } : { targets: { ice40: { success: true } } })));
      };
    });
  });
  vi.stubGlobal("fetch", fetch);
  return { finish: (fail = false) => { finish(fail); }, fetch };
}

it.each(actions)("%s withdraws old success during rerun and failure", async (action) => {
  let io = transport(action);
  let run = useStudioStore.getState()[action]();
  io.finish(); await run;
  const field = action === "runSynthesis" ? "synthResult" : "multiTargetResult";
  expect(useStudioStore.getState()[field]).not.toBeNull();
  io = transport(action);
  run = useStudioStore.getState()[action]();
  const during = useStudioStore.getState()[field];
  io.finish(true); await run;
  expect(during).toBeNull();
  expect(useStudioStore.getState()[field]).toBeNull();
});

it.each(actions)("%s rejects old success after source invalidation", async (action) => {
  const io = transport(action);
  const run = useStudioStore.getState()[action]();
  useStudioStore.getState().setSourceMode("model");
  io.finish(); await run;
  expect(useStudioStore.getState().synthResult).toBeNull();
  expect(useStudioStore.getState().multiTargetResult).toBeNull();
  expect(useStudioStore.getState().isSimulating).toBe(false);
});

it.each(actions)("%s rejects old failure after target change", async (action) => {
  const io = transport(action);
  const run = useStudioStore.getState()[action]();
  useStudioStore.getState().setSynthTarget("ecp5");
  io.finish(true); await run;
  expect(useStudioStore.getState().error).toBeNull();
});

it.each(actions)("%s refuses another start while busy", async (action) => {
  const io = transport(action);
  const run = useStudioStore.getState()[action]();
  const before = useStudioStore.getState();
  const duplicate = useStudioStore.getState()[action]();
  const after = useStudioStore.getState();
  io.finish();
  expect(after).toBe(before);
  await run; await duplicate;
});

it.each(actions)("%s remains owned during operator refresh", async (action) => {
  const refresh: ((value: Response) => void)[] = [];
  vi.stubGlobal("fetch", vi.fn<typeof globalThis.fetch>((_url, init) => {
    if (init?.method === "POST") return Promise.resolve(new Response(JSON.stringify(action === "runSynthesis"
      ? { success: true } : { targets: { ice40: { success: true } } })));
    return new Promise<Response>((resolve) => { refresh.push(resolve); });
  }));
  const run = useStudioStore.getState()[action]();
  await vi.waitFor(() => { expect(refresh).toHaveLength(2); });
  expect(useStudioStore.getState().isSimulating).toBe(true);
  useStudioStore.getState().setSynthTarget("ecp5");
  for (const resolve of refresh) resolve(new Response(JSON.stringify({ audit: {}, jobs: [] })));
  await run;
  expect(useStudioStore.getState().synthResult).toBeNull();
  expect(useStudioStore.getState().multiTargetResult).toBeNull();
  expect(useStudioStore.getState().latestSynthesisJobId).toBeNull();
  expect(useStudioStore.getState().latestMultiTargetSynthesisJobId).toBeNull();
  expect(useStudioStore.getState().isSimulating).toBe(false);
});

it.each(actions)("%s preserves currency across an unrelated simulation duration change", async (action) => {
  const io = transport(action);
  const run = useStudioStore.getState()[action]();
  useStudioStore.setState({ duration: 300 });
  io.finish(); await run;
  const field = action === "runSynthesis" ? "synthResult" : "multiTargetResult";
  expect(useStudioStore.getState()[field]).not.toBeNull();
});

it("withdraws both export handles before a refused request", async () => {
  useStudioStore.setState({ verilogSrc: "", latestSynthesisJobId: "old-single", latestMultiTargetSynthesisJobId: "old-multi" });
  const { fetch } = transport("runSynthesis");
  await useStudioStore.getState().runSynthesis();
  expect(fetch).not.toHaveBeenCalled();
  expect(useStudioStore.getState().latestSynthesisJobId).toBeNull();
  expect(useStudioStore.getState().latestMultiTargetSynthesisJobId).toBeNull();
  expect(useStudioStore.getState().error).toBe("Generate Verilog first");
});
