// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Model-scan nested validation regressions
import { describe, expect, it } from "vitest";

import {
  parseModelBehavior,
  validateModelScanJobReceipt,
  validateModelScanJobResult,
  validateModelScanPollRecord,
} from "./modelScanJobValidation";

const validMetadata = {
  current: 10,
  duration: 100,
  error_count: 0,
  evidence_classification: "analysis" as const,
  failed_models: [] as const,
  input_sha256: "a".repeat(64),
  model_count: 1,
  pattern_counts: { tonic: 1 },
  result_sha256: "b".repeat(64),
  schema_version: "studio.model-scan.v1" as const,
  status: "completed" as const,
};

const validModel = {
  name: "LIFNeuron",
  category: "IF",
  pattern: "tonic",
  description: "tonic",
  rate_hz: 5,
  spike_count: 1,
};

const validResult = {
  models: [validModel],
  scan_metadata: validMetadata,
  schema_version: "studio.model-scan.v1" as const,
};

describe("parseModelBehavior / validateModelScanJobResult", () => {
  it("rejects empty object model entries fail-closed", () => {
    expect(parseModelBehavior({}).ok).toBe(false);
    const rejected = validateModelScanJobResult({
      ...validResult,
      models: [{}],
    });
    expect(rejected.ok).toBe(false);
    if (rejected.ok) {
      return;
    }
    expect(rejected.error).toBe("model_scan_model_name_invalid");
  });

  it("rejects missing metadata digests and counts", () => {
    expect(
      validateModelScanJobResult({
        ...validResult,
        scan_metadata: {
          ...validMetadata,
          input_sha256: "not-hex",
        },
      }).ok,
    ).toBe(false);
    expect(
      validateModelScanJobResult({
        ...validResult,
        scan_metadata: {
          ...validMetadata,
          model_count: "1",
        },
      }).ok,
    ).toBe(false);
    expect(
      validateModelScanJobResult({
        ...validResult,
        scan_metadata: {
          evidence_classification: "analysis",
          schema_version: "studio.model-scan.v1",
          status: "completed",
        },
      }).ok,
    ).toBe(false);
  });

  it("rejects invalid failed_models and pattern_counts entries", () => {
    expect(
      validateModelScanJobResult({
        ...validResult,
        scan_metadata: {
          ...validMetadata,
          failed_models: [{ name: "X" }],
        },
      }).ok,
    ).toBe(false);
    expect(
      validateModelScanJobResult({
        ...validResult,
        scan_metadata: {
          ...validMetadata,
          pattern_counts: { tonic: "many" },
        },
      }).ok,
    ).toBe(false);
    expect(
      validateModelScanJobResult({
        ...validResult,
        scan_metadata: {
          ...validMetadata,
          pattern_counts: null,
        },
      }).ok,
    ).toBe(false);
  });

  it("accepts a fully-typed valid result", () => {
    const ok = validateModelScanJobResult(validResult);
    expect(ok.ok).toBe(true);
    if (!ok.ok) {
      return;
    }
    expect(ok.value.models[0]?.name).toBe("LIFNeuron");
    expect(ok.value.scan_metadata.input_sha256).toHaveLength(64);
  });
});

describe("receipt and poll job binding", () => {
  it("rejects receipt job id/kind mismatches", () => {
    const base = {
      execution_mode: "async_job",
      job_id: "sj_scan_1",
      schema_version: "studio.model-scan.job.v1",
      status_route: "/api/studio/jobs/sj_scan_1",
      job: {
        job_id: "sj_scan_1",
        kind: "model_scan",
        status: "pending",
      },
    };
    expect(validateModelScanJobReceipt(base).ok).toBe(true);
    expect(
      validateModelScanJobReceipt({
        ...base,
        job: { ...base.job, kind: "analysis" },
      }).ok,
    ).toBe(false);
    expect(
      validateModelScanJobReceipt({
        ...base,
        job: { ...base.job, job_id: "sj_other" },
      }).ok,
    ).toBe(false);
  });

  it("rejects poll job id/kind mismatches against retained session id", () => {
    expect(
      validateModelScanPollRecord(
        { job_id: "sj_scan_1", kind: "model_scan", status: "running" },
        "sj_scan_1",
      ).ok,
    ).toBe(true);
    expect(
      validateModelScanPollRecord(
        { job_id: "sj_other", kind: "model_scan", status: "running" },
        "sj_scan_1",
      ).ok,
    ).toBe(false);
    expect(
      validateModelScanPollRecord(
        { job_id: "sj_scan_1", kind: "analysis", status: "running" },
        "sj_scan_1",
      ).ok,
    ).toBe(false);
    expect(
      validateModelScanPollRecord(
        { job_id: "sj_scan_1", kind: "model_scan", status: "running" },
        null,
      ).ok,
    ).toBe(false);
  });
});
