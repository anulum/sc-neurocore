// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Compile transport ownership through public actions

import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { ModelDetail } from "../api/client";
import { useStudioStore } from "./studio";
import { studioGuidedFlowInputs } from "../studioGuidedFlowInputs";

const initial = useStudioStore.getState();
const actions = ["runCompile", "runCosim"] as const;
type Action = typeof actions[number];
const trace = {
  schema_version: "studio.compile-traceability.v1", status: "completed",
  evidence_classification: "compile", source: "model", input_sha256: "a".repeat(64),
  output: { language: "verilog", module_name: "neuron", rtl_chars: 24, rtl_sha256: "b".repeat(64) },
  source_payload: {}, traceability_sha256: "c".repeat(64),
};
const detail: ModelDetail = {
  module: "controlled", category: "point", tier: 0, evidence_kind: "controlled",
  science_tier: 0, science_label: "S0", silicon_tier: null, silicon_label: "none",
  verified_science_tier: 0, verified_science_label: "S0", verified_silicon_tier: null,
  verified_silicon_label: "none", verified_profile: null, is_perfect_verified: false,
  validation_metric: "none", integration_method: "map", terminal_silicon_tier: "",
  terminal_reason: "Controlled fixture, not qualification evidence", category_slug: "point",
  category_source: "declared", metadata_state: "available", metadata_error: null,
  identity_kind: "source-literature", counts_in_source_catalogue: true,
  public_label: "", aliases: [], family: "point", maturity: "experimental",
  biophysical_detail: "point", n_params: 0, n_state_vars: 0, state_var_names: [],
  dt: 0.1, description: "Controlled transport case", intended_use: [], hardware_fit: [],
  behavior_tags: [], provenance: null, docstring: "Controlled transport case",
  display_name: "Controlled", dynamics: {}, backends: [], documentation_slug: "controlled",
  reproducibility: { reference_config: "", golden_trace_sha256: "", reproducible: false },
  name: "AdaptiveThresholdIFNeuron", params: [], state_vars: [],
  compile_configuration: { schema_name: "adaptive_threshold_if", default_integrator: "map",
    integrators: ["map"], cosim_integrators: ["map"], default_q_format: "Q8.8", q_formats: ["Q8.8", "Q16.16"], numeric_contracts: {} },
};

beforeEach(() => { useStudioStore.setState({ sourceMode: "model", selectedModelName: detail.name, modelDetail: detail, modelIntegrator: "map" }); });
afterEach(() => { useStudioStore.setState(initial, true); vi.unstubAllGlobals(); });

/**
 * Hold the HTTP boundary while state changes; payloads are not silicon evidence.
 *
 * @param action - Compile surface being exercised.
 * @returns A resolver for the pending transport.
 */
function deferred(action: Action): (fail?: boolean) => void {
  let finish = (_fail?: boolean): void => { throw new Error("request absent"); };
  vi.stubGlobal("fetch", vi.fn<typeof globalThis.fetch>(() => new Promise<Response>((resolve, reject) => {
    finish = (fail = false) => {
      if (fail) reject(new Error("worker failed"));
      else resolve(new Response(JSON.stringify(action === "runCompile"
        ? { verilog: "module neuron; endmodule", chars: 24, module_name: "neuron", compile_traceability: trace }
        : { bit_exact: true, rtl: { source_sha256: "b".repeat(64) } })));
    };
  })));
  return (fail) => { finish(fail); };
}

it.each(actions)("%s drops an obsolete success", async (action) => {
  const finish = deferred(action);
  const pending = useStudioStore.getState()[action]();
  useStudioStore.getState().setModelQFormat("Q16.16");
  finish(); await pending;
  expect(useStudioStore.getState().compileTraceability).toBeNull();
  expect(useStudioStore.getState().cosimResult).toBeNull();
  expect(useStudioStore.getState().isSimulating).toBe(false);
});

it.each(actions)("%s drops an obsolete failure", async (action) => {
  const finish = deferred(action);
  const pending = useStudioStore.getState()[action]();
  useStudioStore.getState().setModelQFormat("Q16.16");
  finish(true); await pending;
  expect(useStudioStore.getState().error).toBeNull();
  expect(useStudioStore.getState().isSimulating).toBe(false);
});

it.each(actions)("%s withdraws previous success while rerunning and after failure", async (action) => {
  let finish = deferred(action);
  let pending = useStudioStore.getState()[action]();
  finish(); await pending;
  const field = action === "runCompile" ? "compileTraceability" : "cosimResult";
  expect(useStudioStore.getState()[field]).not.toBeNull();
  finish = deferred(action);
  pending = useStudioStore.getState()[action]();
  const during = useStudioStore.getState()[field];
  finish(true); await pending;
  expect(during).toBeNull();
  expect(useStudioStore.getState()[field]).toBeNull();
});

it.each(actions)("%s refuses a duplicate while busy", async (action) => {
  const finish = deferred(action);
  const pending = useStudioStore.getState()[action]();
  const before = useStudioStore.getState();
  const duplicate = useStudioStore.getState()[action]();
  const after = useStudioStore.getState();
  finish();
  expect(after).toBe(before);
  await pending; await duplicate;
});

it.each([false, true])("ODE compile rejects obsolete outcomes, failure=%s", async (fail) => {
  useStudioStore.setState({ sourceMode: "ode", equations: ["dv/dt = -v"] });
  const finish = deferred("runCompile");
  const pending = useStudioStore.getState().runCompile();
  useStudioStore.setState({ equations: ["dv/dt = v"] });
  finish(fail); await pending;
  expect(useStudioStore.getState().compileTraceability).toBeNull();
  expect(useStudioStore.getState().error).toBeNull();
});

it("accepts ODE compilation despite unrelated duration changes", async () => {
  useStudioStore.setState({ sourceMode: "ode", equations: ["dv/dt = -v"] });
  const finish = deferred("runCompile");
  const pending = useStudioStore.getState().runCompile();
  useStudioStore.setState({ duration: 500 });
  finish(); await pending;
  expect(useStudioStore.getState().verilogSrc).toBe("module neuron; endmodule");
});

it.each(actions)("%s respects its actual current dependency", async (action) => {
  const finish = deferred(action);
  const pending = useStudioStore.getState()[action]();
  useStudioStore.setState({ current: 99 });
  finish(); await pending;
  if (action === "runCompile") expect(useStudioStore.getState().compileTraceability).not.toBeNull();
  else expect(useStudioStore.getState().cosimResult).toBeNull();
});

it.each(actions)("%s reports invalid changed configuration and releases busy state", async (action) => {
  const finish = deferred(action);
  const pending = useStudioStore.getState()[action]();
  useStudioStore.setState({ modelQFormat: "invalid" });
  finish(); await pending;
  expect(useStudioStore.getState().error).toContain("not offered");
  expect(useStudioStore.getState().isSimulating).toBe(false);
});

it("refuses unsupported parity without an HTTP call", async () => {
  useStudioStore.setState({ sourceMode: "ode" });
  const fetch = vi.fn<typeof globalThis.fetch>();
  vi.stubGlobal("fetch", fetch);
  await useStudioStore.getState().runCosim();
  expect(fetch).not.toHaveBeenCalled();
  expect(useStudioStore.getState().error).toContain("requires catalogue model");
});

it.each(["rerun", "success", "current"] as const)("withdraws model synthesis qualification on parity %s", async (change) => {
  const compiled = deferred("runCompile");
  const compile = useStudioStore.getState().runCompile();
  compiled(); await compile;
  const oldTrace = useStudioStore.getState().compileTraceability;
  useStudioStore.setState({ synthResult: { success: true, silicon_terminal: { success: true } } as NonNullable<typeof initial.synthResult>,
    latestSynthesisJobId: "sj_previous" });
  const finish = deferred("runCosim");
  const pending = change !== "current" ? useStudioStore.getState().runCosim() : Promise.resolve();
  if (change === "current") useStudioStore.getState().setCurrent(99);
  const during = useStudioStore.getState();
  if (change !== "current") finish(change === "rerun");
  await pending;
  expect(studioGuidedFlowInputs(during, { trainingSkipped: false, evidenceExportSatisfied: false }).synthesisComplete).toBe(false);
  expect(useStudioStore.getState().latestSynthesisJobId).toBeNull();
  expect(useStudioStore.getState().synthResult).toBeNull();
  expect(useStudioStore.getState().compileTraceability).toBe(oldTrace);
  expect(useStudioStore.getState().verilogSrc).toBe("module neuron; endmodule");
  if (change === "success") expect(useStudioStore.getState().cosimResult?.bit_exact).toBe(true);
});

it("preserves ODE synthesis when only simulation current changes", () => {
  const synthesis = { success: true } as NonNullable<typeof initial.synthResult>;
  useStudioStore.setState({ sourceMode: "ode", synthResult: synthesis, latestSynthesisJobId: "sj_ode" });
  useStudioStore.getState().setCurrent(99);
  expect(useStudioStore.getState().synthResult).toBe(synthesis);
  expect(useStudioStore.getState().latestSynthesisJobId).toBe("sj_ode");
});

it("withdraws compiled output immediately when resetting model defaults", async () => {
  useStudioStore.setState({ modelParams: { changed_parameter: 42 }, dt: 0.5 });
  const compiled = deferred("runCompile");
  const compile = useStudioStore.getState().runCompile();
  compiled(); await compile;
  expect(useStudioStore.getState().compileTraceability).not.toBeNull();
  const fetch = vi.fn<typeof globalThis.fetch>().mockRejectedValue(new Error("Reset simulation failed"));
  vi.stubGlobal("fetch", fetch);
  useStudioStore.getState().resetDefaults();
  const immediately = useStudioStore.getState();
  await vi.waitFor(() => { expect(useStudioStore.getState().isSimulating).toBe(false); });
  expect(immediately.modelParams).toEqual({});
  expect(immediately.dt).toBe(detail.dt);
  expect(immediately.compileTraceability).toBeNull();
  expect(immediately.verilogSrc).toBe("");
  expect(useStudioStore.getState().error).toBe("Reset simulation failed");
  expect(useStudioStore.getState().compileTraceability).toBeNull();
  expect(fetch).toHaveBeenCalledOnce();
});
