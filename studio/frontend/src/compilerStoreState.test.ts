// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio compiler store state helper tests
import { describe, expect, it } from "vitest";

import type {
  CompileResponse,
  CompileTraceability,
  ModelCosimReport,
  IRBuildResponse,
  SVDirectResponse,
  SVEmitResponse,
} from "./api/client";
import {
  compilerErrorState,
  compilerConfigurationInvalidatedState,
  compilerCosimInvalidatedState,
  compilerCosimLoadedState,
  compilerFailureState,
  compilerIRLoadedState,
  compilerRunStartState,
  compilerSVDirectLoadedState,
  compilerSVLoadedState,
  compilerVerilogLoadedState,
} from "./compilerStoreState";

function compileTraceability(
  overrides: Partial<CompileTraceability> = {},
): CompileTraceability {
  return {
    evidence_classification: overrides.evidence_classification ?? "compile",
    input_sha256: overrides.input_sha256 ?? "1".repeat(64),
    output: overrides.output ?? {
      language: "systemverilog",
      module_name: "lif_neuron",
      rtl_chars: 128,
      rtl_sha256: "2".repeat(64),
    },
    schema_version: overrides.schema_version ?? "studio.compile-traceability.v1",
    source: overrides.source ?? "ode",
    source_payload: overrides.source_payload ?? {
      equations: ["dv/dt = -(v - E_L) / tau_m + I / C"],
      init: { v: -65 },
      params: { C: 1, E_L: -65, tau_m: 10 },
      reset: "v = -65",
      threshold: "v > -50",
    },
    status: overrides.status ?? "completed",
    traceability_sha256: overrides.traceability_sha256 ?? "3".repeat(64),
  };
}

function compileResponse(overrides: Partial<CompileResponse> = {}): CompileResponse {
  return {
    chars: overrides.chars ?? 128,
    compile_traceability: overrides.compile_traceability ?? compileTraceability(),
    module_name: overrides.module_name ?? "lif_neuron",
    verilog: overrides.verilog ?? "module lif_neuron; endmodule",
  };
}

function cosimReport(): ModelCosimReport {
  return {
    bit_exact: true,
    configuration: {
      dt: 0.1,
      integrator: "map",
      model_name: "AdaptiveThresholdIFNeuron",
      q_format: "Q8.8",
      schema_name: "adaptive_threshold_if",
      schema_sha256: "4".repeat(64),
    },
    first_mismatch: null,
    module_name: "sc_adaptive_threshold_if_neuron",
    reference: {
      kind: "generated_bit_true_c",
      source_sha256: "5".repeat(64),
      trace_sha256: "6".repeat(64),
    },
    rtl: {
      kind: "iverilog_vvp",
      source_sha256: "7".repeat(64),
      trace_sha256: "6".repeat(64),
    },
    sample_count: 128,
    schema_version: "studio.cosim-parity.v1",
    signals: ["spike_out", "v_out", "theta_out"],
    status: "completed",
    stimulus: { current: 10, current_q: 2560, n_steps: 128 },
    tools: { gcc: "gcc 13", iverilog: "Icarus 12", vvp: "VVP 12" },
  };
}

function irBuildResponse(overrides: Partial<IRBuildResponse> = {}): IRBuildResponse {
  return {
    errors: overrides.errors ?? [],
    graph_name: overrides.graph_name ?? "lif_graph",
    ir_text: overrides.ir_text ?? "%0 = input current",
    n_inputs: overrides.n_inputs ?? 1,
    n_ops: overrides.n_ops ?? 3,
    n_outputs: overrides.n_outputs ?? 1,
    params_q88: overrides.params_q88 ?? { tau_m: 2560 },
  };
}

function svEmitResponse(overrides: Partial<SVEmitResponse> = {}): SVEmitResponse {
  return {
    chars: overrides.chars ?? 96,
    graph_name: overrides.graph_name ?? "lif_graph",
    systemverilog: overrides.systemverilog ?? "module lif_graph; endmodule",
  };
}

function svDirectResponse(overrides: Partial<SVDirectResponse> = {}): SVDirectResponse {
  return {
    chars: overrides.chars ?? 160,
    compile_traceability: overrides.compile_traceability ?? compileTraceability(),
    ir_repr: overrides.ir_repr ?? "%0 = input current\n%1 = lif %0",
    module_name: overrides.module_name ?? "lif_neuron",
    verilog: overrides.verilog ?? "module lif_neuron; endmodule",
  };
}

describe("compiler store state helpers", () => {
  it("builds mode errors, start patches, and failures", () => {
    expect(compilerErrorState("IR build requires ODE mode")).toEqual({
      error: "IR build requires ODE mode",
    });
    expect(compilerRunStartState("verilog")).toEqual({
      activeTab: "verilog",
      error: null,
      isSimulating: true,
    });
    expect(compilerRunStartState("ir")).toEqual({
      activeTab: "ir",
      error: null,
      isSimulating: true,
    });
    expect(compilerFailureState(new Error("compiler unavailable"))).toEqual({
      error: "compiler unavailable",
      isSimulating: false,
    });
  });

  it("builds Verilog compile result patches", () => {
    const response = compileResponse();

    expect(compilerVerilogLoadedState(response)).toEqual({
      compileTraceability: response.compile_traceability,
      isSimulating: false,
      verilogSrc: "module lif_neuron; endmodule",
    });
  });

  it("loads co-simulation parity and invalidates stale hardware evidence", () => {
    const report = cosimReport();

    expect(compilerCosimLoadedState(report)).toEqual({
      cosimResult: report,
      isSimulating: false,
    });
    expect(compilerCosimInvalidatedState()).toEqual({ cosimResult: null });
    expect(compilerConfigurationInvalidatedState()).toEqual({
      compileTraceability: null,
      cosimResult: null,
      multiTargetResult: null,
      synthResult: null,
      verilogSrc: "",
    });
  });

  it("builds IR and emitted SystemVerilog result patches", () => {
    const irResponse = irBuildResponse({ errors: ["unreachable output"] });
    const svResponse = svEmitResponse();

    expect(compilerIRLoadedState(irResponse)).toEqual({
      irErrors: ["unreachable output"],
      irText: "%0 = input current",
      isSimulating: false,
    });
    expect(compilerSVLoadedState(svResponse)).toEqual({
      svSource: "module lif_graph; endmodule",
    });
  });

  it("builds direct SystemVerilog compile result patches", () => {
    const response = svDirectResponse();

    expect(compilerSVDirectLoadedState(response)).toEqual({
      compileTraceability: response.compile_traceability,
      irText: "%0 = input current\n%1 = lif %0",
      isSimulating: false,
      svSource: "module lif_neuron; endmodule",
    });
  });
});
