// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio simulation input state helper tests
import { describe, expect, it } from "vitest";

import type { ModelDetail } from "./api/client";
import {
  activeTabState,
  currentState,
  dtState,
  durationState,
  equationsState,
  modelDefaultsState,
  modelFilterState,
  numberRecordEntryState,
  protocolState,
  resetState,
  sourceModeState,
  sweepParamState,
  sweepParamYState,
  thresholdState,
} from "./studioInputState";

function modelDetail(overrides: Partial<ModelDetail> = {}): ModelDetail {
  return {
    category: overrides.category ?? "point",
    docstring: overrides.docstring ?? "Leaky integrate-and-fire neuron.",
    dt: overrides.dt ?? 0.05,
    module: overrides.module ?? "lif",
    name: overrides.name ?? "lif",
    params: overrides.params ?? [
      { default: 10, name: "tau_m" },
      { default: -65, name: "E_L" },
    ],
    state_vars: overrides.state_vars ?? [{ default: -65, name: "v" }],
  };
}

describe("Studio input state helpers", () => {
  it("builds scalar and tab state patches", () => {
    expect(sourceModeState("ode")).toEqual({ sourceMode: "ode" });
    expect(thresholdState("v > -50")).toEqual({ threshold: "v > -50" });
    expect(resetState("v = -65")).toEqual({ reset: "v = -65" });
    expect(dtState(0.05)).toEqual({ dt: 0.05 });
    expect(durationState(250)).toEqual({ duration: 250 });
    expect(currentState(8)).toEqual({ current: 8 });
    expect(protocolState("step")).toEqual({ protocol: "step" });
    expect(activeTabState("phase")).toEqual({ activeTab: "phase" });
    expect(modelFilterState("conductance")).toEqual({ modelFilter: "conductance" });
    expect(sweepParamState("tau_m")).toEqual({ sweepParam: "tau_m" });
    expect(sweepParamYState("C")).toEqual({ sweepParamY: "C" });
  });

  it("copies equation arrays when building equation patches", () => {
    const equations = ["dv/dt = -v / tau"];
    const patch = equationsState(equations);

    expect(patch).toEqual({ equations: ["dv/dt = -v / tau"] });
    expect(patch.equations).not.toBe(equations);
  });

  it("updates number-record entries without mutating the current record", () => {
    const params = { tau_m: 10 };

    expect(numberRecordEntryState("odeParams", params, "E_L", -65)).toEqual({
      odeParams: { E_L: -65, tau_m: 10 },
    });
    expect(params).toEqual({ tau_m: 10 });
    expect(numberRecordEntryState("modelParams", {}, "v", -65)).toEqual({
      modelParams: { v: -65 },
    });
    expect(numberRecordEntryState("odeInit", {}, "w", 0)).toEqual({
      odeInit: { w: 0 },
    });
  });

  it("builds model default reset patches from model details", () => {
    expect(modelDefaultsState(modelDetail())).toEqual({
      current: 10,
      dt: 0.05,
      duration: 100,
      modelParams: {
        E_L: -65,
        tau_m: 10,
        v: -65,
      },
    });
  });
});
