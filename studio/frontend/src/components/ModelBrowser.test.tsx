import { describe, expect, it } from "vitest";

import type { ModelScanMetadata } from "../api/client";
import { buildModelScanEvidenceItems } from "./ModelBrowser";

const scanMetadata: ModelScanMetadata = {
  current: 10,
  duration: 100,
  evidence_classification: "analysis",
  input_sha256: "1".repeat(64),
  model_count: 118,
  pattern_counts: { bursting: 7, silent: 20, tonic: 91 },
  result_sha256: "2".repeat(64),
  schema_version: "studio.model-scan.v1",
  status: "completed",
};

describe("ModelBrowser", () => {
  it("builds path-free model-scan evidence labels", () => {
    expect(buildModelScanEvidenceItems(scanMetadata)).toEqual([
      { label: "class", value: "analysis" },
      { label: "status", value: "completed" },
      { label: "models", value: "118" },
      { label: "in", value: "1111111111" },
      { label: "out", value: "2222222222" },
    ]);
  });

  it("omits evidence labels before a scan completes", () => {
    expect(buildModelScanEvidenceItems(null)).toEqual([]);
  });
});
