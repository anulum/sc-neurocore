// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Native compiler admission and invalidation through public actions

import { afterEach, beforeEach, expect, it } from "vitest";
import { defaultEquationIrConfiguration } from "../equationIrConfig";
import { nativeSVEmitResult } from "../equationIrTestSupport";
import { useStudioStore } from "./studio";

const initial = useStudioStore.getState();
const actions = ["runBuildIR", "runEmitSV", "runVerifyIR", "runReplayIR"] as const;
beforeEach(() => { useStudioStore.setState({ sourceMode: "ode", equations: ["dv/dt = I"],
  odeParams: {}, odeInit: { v: 0 }, equationIrConfig: defaultEquationIrConfiguration() }); });
afterEach(() => { useStudioStore.setState(initial, true); });

it.each(actions)("%s refuses an incomplete compiler draft before execution", async (action) => {
  const config = defaultEquationIrConfiguration(); config.profile.data_width = "";
  useStudioStore.getState().setEquationIrConfig(config);
  await useStudioStore.getState()[action]();
  const state = useStudioStore.getState();
  expect(state.error).toBe("Total bits must be an exact integer of at least 1.");
  expect(state.irResult).toBeNull(); expect(state.svSource).toBe("");
  expect(state.isSimulating).toBe(false);
});

it.each(actions)("%s refuses catalogue mode without producing equation artifacts", async (action) => {
  useStudioStore.setState({ sourceMode: "model" });
  await useStudioStore.getState()[action]();
  const state = useStudioStore.getState();
  expect(state.error).toBe("IR/SV generation requires ODE mode");
  expect(state.irText).toBe(""); expect(state.irResult).toBeNull();
  expect(state.svSource).toBe(""); expect(state.isSimulating).toBe(false);
});

it.each(actions)("%s preserves the current operation when already busy", async (action) => {
  useStudioStore.setState({ isSimulating: true, irText: "work in progress" });
  const before = useStudioStore.getState();
  await before[action]();
  expect(useStudioStore.getState()).toBe(before);
});

it.each(["runVerifyIR", "runReplayIR"] as const)("%s refuses absent current IR", async (action) => {
  await useStudioStore.getState()[action]();
  expect(useStudioStore.getState().error).toBe("Build IR for the current equation inputs before inspecting it.");
  expect(useStudioStore.getState().isSimulating).toBe(false);
});

it("editing settings withdraws native source, replay and downstream hardware results", () => {
  useStudioStore.setState({ irResult: nativeSVEmitResult(), irRequestKey: "old request",
    irText: nativeSVEmitResult().ir_text, svSource: nativeSVEmitResult().systemverilog,
    verilogSrc: "old direct RTL", latestSynthesisJobId: "sj_old_synthesis" });
  const config = defaultEquationIrConfiguration();
  config.model.rng_seed = "18446744073709551615";
  useStudioStore.getState().setEquationIrConfig(config);
  const state = useStudioStore.getState();
  expect(state.equationIrConfig.model.rng_seed).toBe("18446744073709551615");
  expect(state.irResult).toBeNull(); expect(state.irRequestKey).toBeNull();
  expect(state.irText).toBe(""); expect(state.svSource).toBe(""); expect(state.verilogSrc).toBe("");
  expect(state.latestSynthesisJobId).toBeNull(); expect(state.synthResult).toBeNull();
});
