// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Compiler Inspector rendering against the real Studio store
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { useStudioStore } from "../stores/studio";
import { defaultEquationIrConfiguration } from "../equationIrConfig";
import { nativeIRBuildResult, nativeIRInvalidResult, nativeSVEmitResult } from "../equationIrTestSupport";
import { compilerIRLoadedState } from "../compilerStoreState";
import CompilerInspector from "./CompilerInspector";

const initial = useStudioStore.getState();
let root: Root;
let container: HTMLDivElement;
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
beforeEach(() => {
  container = document.createElement("div");
  document.body.append(container);
  act(() => {
    useStudioStore.setState({ sourceMode: "ode", equationIrConfig: defaultEquationIrConfiguration(),
      equations: ["dv/dt = I"], odeInit: { v: 0 }, odeParams: {}, dt: 0.5,
      irResult: null, irText: "", irRequestKey: null, svSource: "", verilogSrc: "" });
    root = createRoot(container);
    root.render(<CompilerInspector />);
  });
});
afterEach(() => {
  act(() => { root.unmount(); useStudioStore.setState(initial, true); });
  container.remove();
});

describe("CompilerInspector", () => {
  it("separates graph structure, source replay and hardware qualification", () => {
    expect(container.textContent).toContain("hardware validation are separate results");
    act(() => { useStudioStore.setState({ irResult: nativeIRBuildResult(),
      irText: nativeIRBuildResult().ir_text, irRequestKey: "captured request" }); });
    expect(container.textContent).toContain("Graph structure valid");
    expect(container.textContent).toContain("Source replay not run");
    expect(container.textContent).toContain("30 operations");
    expect(container.textContent).toContain("1 clocks per timestep");
    expect(container.textContent).toContain("v_out: 16 bits");
    act(() => { useStudioStore.setState({ irResult: nativeSVEmitResult(),
      svSource: nativeSVEmitResult().systemverilog }); });
    expect(container.textContent).toContain("Source replay passed");
    expect(container.querySelector('[aria-label="Generated SystemVerilog"]')?.textContent)
      .toContain("module sc_equation_neuron");
    expect(container.querySelector('[aria-label="Native intermediate representation"]')?.textContent)
      .toContain("sc.graph @sc_equation_neuron");
  });

  it("withdraws native validation and export eligibility after settings change", () => {
    act(() => { useStudioStore.setState({ irResult: nativeSVEmitResult(),
      irText: nativeIRBuildResult().ir_text, irRequestKey: "captured request",
      svSource: nativeSVEmitResult().systemverilog }); });
    const config = defaultEquationIrConfiguration(); config.profile.fraction = "7";
    act(() => { useStudioStore.getState().setEquationIrConfig(config); });
    expect(container.textContent).not.toContain("Source replay passed");
    const replay = [...container.querySelectorAll("button")].find((button) => button.textContent === "Replay source");
    expect(replay?.disabled).toBe(true);
    expect(useStudioStore.getState().svSource).toBe("");
  });

  it("prevents ODE compilation in catalogue mode and honours running work", () => {
    act(() => { useStudioStore.setState({ sourceMode: "model" }); });
    for (const label of ["Build IR", "Emit SV"]) {
      expect([...container.querySelectorAll("button")].find((button) => button.textContent === label)?.disabled).toBe(true);
    }
    act(() => { useStudioStore.setState({ sourceMode: "ode", isSimulating: true }); });
    expect(container.textContent).toContain("Compiling");
    expect([...container.querySelectorAll("button")].find((button) => button.textContent === "Build IR")?.disabled).toBe(true);
  });

  it("shows native structural diagnostics without claiming source replay or an output contract", () => {
    const rejected = nativeIRInvalidResult();
    act(() => { useStudioStore.setState(compilerIRLoadedState(rejected, "inspected graph")); });
    expect(container.textContent).toContain("Graph structure invalid");
    expect(container.textContent).toContain("Source replay unavailable");
    expect(container.querySelector('[role="alert"]')?.textContent).toContain("reserved generated namespace");
    expect(container.textContent).not.toContain("clocks per timestep");
    expect(container.textContent).not.toContain("Output widths:");
  });
});
