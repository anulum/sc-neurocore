// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — editable hosted scientific examples

import type { HostOperation } from "./hostApiTypes";

/** User-facing scientific action names, independent of the message transport. */
export const HOST_OPERATION_LABELS: Record<HostOperation, string> = {
  simulate: "Simulate", fi_curve: "Firing-rate curve", bifurcation: "Numerical extrema sweep",
  heatmap: "Response heatmap", sensitivity: "Parameter sensitivity", "ir.build": "Build native graph",
  "ir.verify": "Verify graph structure", "ir.replay": "Replay bound source", "ir.emit-sv": "Emit SystemVerilog",
};

/**
 * Supply editable scientific inputs, reusing the actual prior graph when present.
 *
 * @param operation - Selected supported action.
 * @param irText - Genuine graph from a prior accepted compiler result.
 * @returns JSON request text; examples are inputs, never fabricated results.
 */
export function hostedExample(operation: HostOperation, irText: string): string {
  const base = { equations: ["dv/dt = (I-v+offset)/tau"], params: { tau: 10, offset: 0 },
    init: { v: 0 }, dt: 0.1, duration: 20 };
  let request: object;
  switch (operation) {
    case "simulate": request = { ...base, current: 10, protocol: "constant" }; break;
    case "fi_curve": request = { ...base, i_min: 0, i_max: 20, i_steps: 5 }; break;
    case "bifurcation": request = { ...base, current: 10, sweep_param: "tau", sweep_min: 5,
      sweep_max: 20, sweep_steps: 5 }; break;
    case "heatmap": request = { ...base, current: 10, param_x: "tau", x_min: 5, x_max: 20,
      x_steps: 3, param_y: "offset", y_min: 0, y_max: 5, y_steps: 3 }; break;
    case "sensitivity": request = { ...base, current: 10 }; break;
    case "ir.build": request = { schema_version: "studio.equation-ir-request.v1", model: {
      equations: [{ name: "z", expression: "(I-z+a)/tau" }, { name: "a", expression: "z-a" }],
      initial_state: [{ name: "z", value: 3 }, { name: "a", value: 1 }],
      parameters: [{ name: "tau", value: 10 }], constants: [], reset: [], dt: { value: 0.1 },
      method: "gauss_seidel", rng_seed: "18446744073709551615", random_stream: 4294967295,
    }, profile: { module_name: "hosted_equations", data_width: 32, fraction: 16,
      signed: true, overflow: "saturate", rounding: "nearest", function_lowering: "v2" } }; break;
    case "ir.verify": case "ir.replay": request = { ir_text: irText }; break;
    case "ir.emit-sv": request = { ir_text: irText, source_policy: "require-bound" }; break;
  }
  return JSON.stringify(request, null, 2);
}
