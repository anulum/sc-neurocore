// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — IR and emitted source ownership regressions

import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useStudioStore } from "./studio";

const initial = useStudioStore.getState();
beforeEach(() => { useStudioStore.setState({ sourceMode: "ode", equations: ["dv/dt = -v"] }); });
afterEach(() => { useStudioStore.setState(initial, true); vi.unstubAllGlobals(); });

/**
 * Control the transport, leaving the production client and store wired.
 *
 * @returns FIFO completion handles for individual requests.
 */
function transport() {
  const pending: ((value: object | Error) => void)[] = [];
  const fetch = vi.fn<typeof globalThis.fetch>(() => new Promise<Response>((resolve, reject) => {
    pending.push((value) => {
      if (value instanceof Error) reject(value);
      else resolve(new Response(JSON.stringify(value)));
    });
  }));
  vi.stubGlobal("fetch", fetch);
  return { pending, fetch };
}

it("keeps the IR operation busy until SV emission completes", async () => {
  const { pending } = transport();
  const run = useStudioStore.getState().runBuildIR();
  pending[0]?.({ ir_text: "%0 = input", errors: [] });
  await vi.waitFor(() => { expect(pending).toHaveLength(2); });
  const busy = useStudioStore.getState().isSimulating;
  pending[1]?.({ systemverilog: "module current; endmodule" });
  await run;
  expect(busy).toBe(true);
  expect(useStudioStore.getState().svSource).toContain("current");
  expect(useStudioStore.getState().isSimulating).toBe(false);
});

it.each([false, true])("drops obsolete emitted SV outcomes, failed=%s", async (failed) => {
  const { pending } = transport();
  const run = useStudioStore.getState().runBuildIR();
  pending[0]?.({ ir_text: "%0 = input", errors: [] });
  await vi.waitFor(() => { expect(pending).toHaveLength(2); });
  useStudioStore.getState().setSourceMode("model");
  pending[1]?.(failed ? new Error("old emission") : { systemverilog: "obsolete" });
  await run;
  expect(useStudioStore.getState().svSource).toBe("");
  expect(useStudioStore.getState().irText).toBe("");
  expect(useStudioStore.getState().error).toBeNull();
});

it("invalidates both source producers before allowing synthesis", async () => {
  useStudioStore.setState({ svSource: "old SV", verilogSrc: "old RTL", irText: "%0 = stale", irErrors: ["stale"] });
  useStudioStore.getState().setSourceMode("model");
  const { fetch } = transport();
  await useStudioStore.getState().runSynthesis();
  expect(fetch).not.toHaveBeenCalled();
  expect(useStudioStore.getState().error).toBe("Generate Verilog first");
  expect(useStudioStore.getState().svSource).toBe("");
  expect(useStudioStore.getState().irText).toBe("");
  expect(useStudioStore.getState().irErrors).toEqual([]);
});

it.each([false, true])("drops obsolete direct SV outcomes, failed=%s", async (failed) => {
  const { pending } = transport();
  const run = useStudioStore.getState().runEmitSV();
  useStudioStore.getState().setSourceMode("model");
  pending[0]?.(failed ? new Error("old emission") : { verilog: "old SV", ir_repr: "old IR" });
  await run;
  expect(useStudioStore.getState().svSource).toBe("");
  expect(useStudioStore.getState().error).toBeNull();
});

it("sends newly compiled Verilog to synthesis instead of older emitted SV", async () => {
  const { pending, fetch } = transport();
  let run = useStudioStore.getState().runEmitSV();
  pending[0]?.({ verilog: "old SV", ir_repr: "old IR", compile_traceability: {} });
  await run;
  run = useStudioStore.getState().runCompile();
  pending[1]?.({ verilog: "new RTL", compile_traceability: {} });
  await run;
  run = useStudioStore.getState().runSynthesis();
  const body = fetch.mock.calls[2]?.[1]?.body;
  if (typeof body !== "string") throw new Error("Expected JSON synthesis body");
  const submitted: unknown = JSON.parse(body);
  pending[2]?.(new Error("controlled synthesis stop"));
  await run;
  expect(submitted).toMatchObject({ verilog: "new RTL" });
  expect(useStudioStore.getState().svSource).toBe("");
});

it("does not emit SV after IR validation errors", async () => {
  const { pending, fetch } = transport();
  const run = useStudioStore.getState().runBuildIR();
  pending[0]?.({ ir_text: "invalid IR", errors: ["invalid operand"] });
  await run;
  expect(fetch).toHaveBeenCalledOnce();
  expect(useStudioStore.getState().irErrors).toEqual(["invalid operand"]);
  expect(useStudioStore.getState().svSource).toBe("");
  expect(useStudioStore.getState().isSimulating).toBe(false);
});

it.each(["runBuildIR", "runEmitSV"] as const)("%s refuses duplicates and withdraws sources on failure", async (action) => {
  useStudioStore.setState({ svSource: "old", irText: "old", verilogSrc: "old" });
  const { pending, fetch } = transport();
  const run = useStudioStore.getState()[action]();
  await useStudioStore.getState()[action]();
  pending[0]?.(new Error("generation failed"));
  await run;
  expect(fetch).toHaveBeenCalledOnce();
  expect(useStudioStore.getState().error).toBe("generation failed");
  expect(useStudioStore.getState().svSource).toBe("");
  expect(useStudioStore.getState().verilogSrc).toBe("");
  expect(useStudioStore.getState().irText).toBe("");
  expect(useStudioStore.getState().isSimulating).toBe(false);
});
