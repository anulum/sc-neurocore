// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Co-simulation verdict tests

import { describe, expect, it } from "vitest";

import type { ModelCosimReport } from "./api/client";
import { cosimVerdict } from "./cosimVerdict";

/**
 * A completed co-simulation report of the shape the backend serves.
 *
 * @param overrides - Fields to change.
 * @returns The report.
 */
function report(overrides: Partial<ModelCosimReport> = {}): ModelCosimReport {
  return {
    bit_exact: true,
    configuration: {
      dt: 0.1, integrator: "euler", model_name: "AdExNeuron", q_format: "Q16.16",
      schema_name: "adex", schema_sha256: "a".repeat(64),
    },
    first_mismatch: null,
    module_name: "sc_adex_neuron",
    reference: { kind: "generated_bit_true_c", source_sha256: "b".repeat(64), trace_sha256: "c".repeat(64) },
    rtl: { kind: "iverilog_vvp", source_sha256: "d".repeat(64), trace_sha256: "c".repeat(64) },
    sample_count: 128,
    schema_version: "studio.cosim-parity.v1",
    signals: ["spike_out", "v_out"],
    status: "completed",
    stimulus: { current: 10, current_q: 655360, n_steps: 128 },
    tools: { gcc: "gcc 14", iverilog: "12.0", vvp: "12.0" },
    boundary: {
      compared: "rtl_vs_bittrue",
      statement: "the Icarus Verilog RTL against the generated bit-true C kernel",
      not_covered: ["a state landing exactly on the threshold"],
    },
    stress: {
      bit_exact: true,
      first_mismatch: null,
      reference_trace_sha256: "e".repeat(64),
      rtl_trace_sha256: "e".repeat(64),
      sample_count: 56,
      schedule: [{ input_q: -2147483648, steps: 16, reset_before: true }],
    },
    ...overrides,
  };
}

describe("cosimVerdict", () => {
  it("states what agreed and what that does not show", () => {
    const verdict = cosimVerdict(report());
    expect(verdict.pass).toBe(true);
    expect(verdict.label).toBe("RTL = bit-true C kernel");
    expect(verdict.cycles).toBe("128 requested + 56 stress cycles");
    expect(verdict.mismatch).toBeNull();
    expect(verdict.title).toBe(
      "the Icarus Verilog RTL against the generated bit-true C kernel "
      + "Not covered: a state landing exactly on the threshold. "
      + "Not the model's scientific fidelity, and not a proof.",
    );
  });

  it("names a requested-run mismatch before a stress one", () => {
    const mismatch = { cycle: 7, reference: {}, rtl: {}, signals: ["v_out"] };
    const verdict = cosimVerdict(report({ bit_exact: false, first_mismatch: mismatch }));
    expect(verdict.label).toBe("RTL ≠ bit-true C kernel");
    expect(verdict.mismatch).toBe("cycle 7: v_out");
  });

  it("names a mismatch only the stress schedule found", () => {
    const base = report();
    const stress = base.stress && {
      ...base.stress,
      bit_exact: false,
      first_mismatch: { cycle: 17, reference: {}, rtl: {}, signals: ["spike_out", "v_out"] },
    };
    const verdict = cosimVerdict(report({ bit_exact: false, stress }));
    expect(verdict.mismatch).toBe("stress cycle 17: spike_out, v_out");
  });

  it("keeps its caveats for a report without a stress run or boundary", () => {
    const verdict = cosimVerdict(report({ stress: undefined, boundary: undefined }));
    expect(verdict.cycles).toBe("128 cycles");
    expect(verdict.title).toBe(
      "the RTL against the generated bit-true C kernel for the stimulus run "
      + "Not the model's scientific fidelity, and not a proof.",
    );
  });
});
