// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Analysis job nested validation regressions
import { describe, expect, it } from "vitest";

import type { AnalysisJobKind } from "./api/client";
import {
  validateAnalysisJobReceipt,
  validateAnalysisJobResult,
  validateAnalysisPollRecord,
} from "./analysisJobValidation";

/**
 * Build result metadata declaring the given analysis.
 *
 * @param analysisType - The analysis the result claims to be.
 * @param outputKeys - The fields it claims to carry.
 * @returns The metadata.
 */
function metadata(analysisType: AnalysisJobKind, outputKeys: string[]) {
  return {
    analysis_type: analysisType,
    evidence_classification: "analysis" as const,
    input_sha256: "a".repeat(64),
    output_keys: outputKeys,
    result_sha256: "b".repeat(64),
    schema_version: "studio.analysis-result.v1" as const,
    source: "model" as const,
    status: "completed" as const,
  };
}

const fiCurve = {
  analysis_metadata: metadata("fi_curve", ["currents", "rates"]),
  currents: [0, 1, 2],
  rates: [0, 5, 10],
};

const bifurcation = {
  analysis_metadata: metadata("bifurcation", ["param_name", "param_values", "attractors"]),
  attractors: [[1], [2], [3]],
  param_name: "tau_m",
  param_values: [5, 10, 15],
};

const heatmap = {
  analysis_metadata: metadata("heatmap", ["param_x", "rates"]),
  param_x: "tau_m",
  param_y: "R",
  rate_max: 10,
  rate_min: 0,
  rates: [
    [0, 1],
    [2, 3],
  ],
  x_values: [1, 2],
  y_values: [3, 4],
};

const sensitivity = {
  analysis_metadata: metadata("sensitivity", ["base_rate", "sensitivities"]),
  base_rate: 5,
  sensitivities: [
    { param: "tau_m", rate_minus: 4, rate_plus: 6, sensitivity: 0.4 },
  ],
};

describe("validateAnalysisJobResult by kind", () => {
  it("accepts valid completion for all four analysis kinds", () => {
    expect(validateAnalysisJobResult(fiCurve, "fi_curve").ok).toBe(true);
    expect(validateAnalysisJobResult(bifurcation, "bifurcation").ok).toBe(true);
    expect(validateAnalysisJobResult(heatmap, "heatmap").ok).toBe(true);
    expect(validateAnalysisJobResult(sensitivity, "sensitivity").ok).toBe(true);
  });

  it("rejects malformed nested result fields fail-closed", () => {
    expect(
      validateAnalysisJobResult(
        { ...fiCurve, currents: [0, "x"] },
        "fi_curve",
      ).ok,
    ).toBe(false);
    expect(
      validateAnalysisJobResult(
        {
          ...fiCurve,
          analysis_metadata: {
            ...fiCurve.analysis_metadata,
            analysis_type: "heatmap",
          },
        },
        "fi_curve",
      ).ok,
    ).toBe(false);
    expect(
      validateAnalysisJobResult(
        {
          ...fiCurve,
          analysis_metadata: {
            ...fiCurve.analysis_metadata,
            input_sha256: "short",
          },
        },
        "fi_curve",
      ).ok,
    ).toBe(false);
    expect(
      validateAnalysisJobResult(
        { ...heatmap, rates: [[0, 1, 2]] },
        "heatmap",
      ).ok,
    ).toBe(false);
  });
});

/**
 * Build a valid job envelope, for cases to invalidate one field of at a time.
 *
 * @param overrides - The fields to change or remove.
 * @returns The envelope, as parsed JSON rather than as a typed record, since
 *   most cases put something invalid in it.
 */
function jobEnvelope(overrides: Record<string, unknown> = {}) {
  return {
    artifacts: [],
    created_at_utc: "2026-07-19T00:00:00Z",
    error: null,
    execution_model: "thread",
    finished_at_utc: null,
    job_id: "sj_a1",
    kind: "analysis",
    owner: "studio",
    request_id: null,
    result: null,
    started_at_utc: null,
    status: "pending",
    ...overrides,
  };
}

describe("receipt and poll binding", () => {
  const receipt = {
    analysis: "fi_curve",
    execution_mode: "async_job",
    job_id: "sj_a1",
    schema_version: "studio.analysis.job.v1",
    status_route: "/api/studio/jobs/sj_a1",
    job: jobEnvelope(),
  };

  it("rejects receipt analysis/id/kind mismatches", () => {
    expect(validateAnalysisJobReceipt(receipt, "fi_curve").ok).toBe(true);
    expect(validateAnalysisJobReceipt(receipt, "heatmap").ok).toBe(false);
    expect(
      validateAnalysisJobReceipt(
        { ...receipt, job: jobEnvelope({ kind: "model_scan" }) },
        "fi_curve",
      ).ok,
    ).toBe(false);
    expect(
      validateAnalysisJobReceipt(
        { ...receipt, job: jobEnvelope({ job_id: "sj_other" }) },
        "fi_curve",
      ).ok,
    ).toBe(false);
  });

  it("rejects poll id/kind mismatches", () => {
    expect(
      validateAnalysisPollRecord(
        jobEnvelope({ status: "running" }),
        "sj_a1",
      ).ok,
    ).toBe(true);
    expect(
      validateAnalysisPollRecord(
        jobEnvelope({ job_id: "sj_other", status: "running" }),
        "sj_a1",
      ).ok,
    ).toBe(false);
    expect(
      validateAnalysisPollRecord(
        jobEnvelope({ kind: "model_scan", status: "running" }),
        "sj_a1",
      ).ok,
    ).toBe(false);
  });

  it("accepts recovery statuses but refuses an unrecognised status", () => {
    for (const status of ["interrupted", "unknown"]) {
      const parsed = validateAnalysisPollRecord(jobEnvelope({ status }), "sj_a1");
      expect(parsed.ok).toBe(true);
      if (parsed.ok) expect(parsed.value.status).toBe(status);
    }
    expect(validateAnalysisPollRecord(jobEnvelope({ status: "recovered" }), "sj_a1")).toEqual({
      ok: false,
      error: "analysis_poll_job_status_invalid",
    });
  });

  it("rejects fail-open envelopes: empty artifacts, invalid model, nullables, metrics", () => {
    expect(
      validateAnalysisJobReceipt(
        {
          ...receipt,
          job: jobEnvelope({ artifacts: [{}] }),
        },
        "fi_curve",
      ).ok,
    ).toBe(false);
    expect(
      validateAnalysisJobReceipt(
        {
          ...receipt,
          job: jobEnvelope({ execution_model: "fiber" }),
        },
        "fi_curve",
      ).ok,
    ).toBe(false);
    expect(
      validateAnalysisJobReceipt(
        {
          ...receipt,
          job: jobEnvelope({ created_at_utc: 1 }),
        },
        "fi_curve",
      ).ok,
    ).toBe(false);
    expect(
      validateAnalysisJobReceipt(
        {
          ...receipt,
          job: jobEnvelope({ error: 12 }),
        },
        "fi_curve",
      ).ok,
    ).toBe(false);
    expect(
      validateAnalysisJobReceipt(
        {
          ...receipt,
          job: jobEnvelope({ result: "nope" }),
        },
        "fi_curve",
      ).ok,
    ).toBe(false);
    expect(
      validateAnalysisJobReceipt(
        { ...receipt, projected_simulations: "many" },
        "fi_curve",
      ).ok,
    ).toBe(false);
    expect(
      validateAnalysisPollRecord(
        jobEnvelope({ artifacts: [{}], status: "running" }),
        "sj_a1",
      ).ok,
    ).toBe(false);
  });
});
