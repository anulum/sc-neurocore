// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — IR and emitted source ownership regressions

import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useStudioStore } from "./studio";
import { defaultEquationIrConfiguration } from "../equationIrConfig";
import { nativeIRBuildResult, nativeSVEmitResult, nativeIRInvalidResult } from "../equationIrTestSupport";

const initial = useStudioStore.getState();
beforeEach(() => { useStudioStore.setState({ sourceMode: "ode", equations: ["dv/dt = I"], odeParams: {}, odeInit: { v: 0 }, dt: 0.5,
  equationIrConfig: defaultEquationIrConfiguration() }); });
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

it("keeps native SV generation busy across build and emission", async () => {
  const { pending } = transport();
  const run = useStudioStore.getState().runEmitSV();
  pending[0]?.(nativeIRBuildResult());
  await vi.waitFor(() => { expect(pending).toHaveLength(2); });
  const busy = useStudioStore.getState().isSimulating;
  pending[1]?.(nativeSVEmitResult());
  await run;
  expect(busy).toBe(true);
  expect(useStudioStore.getState().svSource).toContain("sc_equation_neuron");
  expect(useStudioStore.getState().isSimulating).toBe(false);
});

it("inspects and emits the current built graph without rebuilding its authored source", async () => {
  const { pending, fetch } = transport();
  const build = useStudioStore.getState().runBuildIR();
  pending[0]?.(nativeIRBuildResult());
  await build;
  const graph = useStudioStore.getState().irText;
  const verify = useStudioStore.getState().runVerifyIR();
  pending[1]?.({ ...nativeIRBuildResult(), operation: "verify", clocks_per_timestep: null,
    pipeline_registers: null, output_widths: null });
  await verify;
  expect(useStudioStore.getState().irResult?.source_replay).toBe("not_requested");
  const replay = useStudioStore.getState().runReplayIR();
  pending[2]?.({ ...nativeIRBuildResult(), operation: "replay", source_replay: "passed" });
  await replay;
  expect(useStudioStore.getState().irResult?.source_replay).toBe("passed");
  const emit = useStudioStore.getState().runEmitSV();
  pending[3]?.(nativeSVEmitResult());
  await emit;
  expect(fetch.mock.calls.map(([url]) => url)).toEqual([
    "/api/ir/build", "/api/ir/verify", "/api/ir/replay", "/api/ir/emit-sv",
  ]);
  expect(useStudioStore.getState().irText).toBe(graph);
  expect(useStudioStore.getState().svSource).toBe(nativeSVEmitResult().systemverilog);
  expect(useStudioStore.getState().isSimulating).toBe(false);
});

it.each([false, true])("does not restore artifacts when a numeric draft becomes incomplete, failed=%s", async (failed) => {
  const { pending } = transport();
  const build = useStudioStore.getState().runBuildIR();
  const draft = defaultEquationIrConfiguration(); draft.profile.fraction = "";
  useStudioStore.getState().setEquationIrConfig(draft);
  pending[0]?.(failed ? new Error("old compiler failure") : nativeIRBuildResult());
  await build;
  expect(useStudioStore.getState().error).toBe("Fractional bits must be an exact integer of at least 0.");
  expect(useStudioStore.getState().irText).toBe("");
  expect(useStudioStore.getState().svSource).toBe("");
  expect(useStudioStore.getState().isSimulating).toBe(false);
});

it.each([false, true])("drops obsolete emitted SV outcomes, failed=%s", async (failed) => {
  const { pending } = transport();
  const run = useStudioStore.getState().runEmitSV();
  pending[0]?.(nativeIRBuildResult());
  await vi.waitFor(() => { expect(pending).toHaveLength(2); });
  useStudioStore.getState().setSourceMode("model");
  pending[1]?.(failed ? new Error("old emission") : nativeSVEmitResult());
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

it.each([false, true])("drops obsolete native build outcomes before SV emission, failed=%s", async (failed) => {
  const { pending } = transport();
  const run = useStudioStore.getState().runEmitSV();
  useStudioStore.getState().setSourceMode("model");
  pending[0]?.(failed ? new Error("old emission") : nativeIRBuildResult());
  expect(pending).toHaveLength(1);
  await run;
  expect(useStudioStore.getState().svSource).toBe("");
  expect(useStudioStore.getState().error).toBeNull();
});

it("sends newly compiled Verilog to synthesis instead of older emitted SV", async () => {
  const { pending, fetch } = transport();
  useStudioStore.setState({ svSource: nativeSVEmitResult().systemverilog });
  const run = useStudioStore.getState().runCompile();
  pending[0]?.({ verilog: "new RTL", compile_traceability: {} });
  await run;
  const synthesis = useStudioStore.getState().runSynthesis();
  const body = fetch.mock.calls[1]?.[1]?.body;
  if (typeof body !== "string") throw new Error("Expected JSON synthesis body");
  const submitted: unknown = JSON.parse(body);
  pending[1]?.(new Error("controlled synthesis stop"));
  await synthesis;
  expect(submitted).toMatchObject({ verilog: "new RTL" });
  expect(useStudioStore.getState().svSource).toBe("");
});

it("does not emit SV after IR validation errors", async () => {
  const { pending, fetch } = transport();
  const run = useStudioStore.getState().runBuildIR();
  pending[0]?.(nativeIRInvalidResult());
  await run;
  expect(fetch).toHaveBeenCalledOnce();
  expect(useStudioStore.getState().irErrors).toEqual(nativeIRInvalidResult().errors);
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
