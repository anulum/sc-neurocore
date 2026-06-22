import { describe, expect, it } from "vitest";

import type { ModelBehavior, ModelScanMetadata } from "../api/client";
import { buildModelScanEvidenceItems, filterAndGroupModels } from "./ModelBrowser";

interface BrowseModel {
  name: string;
  category: string;
  family: string;
  tier: number;
}

const CATALOGUE: BrowseModel[] = [
  { name: "AdExNeuron", category: "Integrate-and-Fire", family: "Integrate-and-Fire", tier: 2 },
  { name: "GLIFNeuron", category: "Integrate-and-Fire", family: "Integrate-and-Fire", tier: 3 },
  { name: "GolgiCell", category: "Cerebellar", family: "Cerebellar", tier: 1 },
  { name: "RulkovMapNeuron", category: "Map-based", family: "Map-based", tier: 3 },
];

const NONE: Record<string, ModelBehavior> = {};

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

  it("groups the catalogue by family with no filters", () => {
    const grouped = filterAndGroupModels(CATALOGUE, {
      modelFilter: "",
      familyFilter: "",
      patternFilter: "",
      minTier: 0,
      behaviors: NONE,
    });
    expect(Object.keys(grouped).sort()).toEqual([
      "Cerebellar",
      "Integrate-and-Fire",
      "Map-based",
    ]);
    expect(grouped["Integrate-and-Fire"].map((m) => m.name)).toEqual([
      "AdExNeuron",
      "GLIFNeuron",
    ]);
  });

  it("restricts the catalogue to a selected family", () => {
    const grouped = filterAndGroupModels(CATALOGUE, {
      modelFilter: "",
      familyFilter: "Cerebellar",
      patternFilter: "",
      minTier: 0,
      behaviors: NONE,
    });
    expect(Object.keys(grouped)).toEqual(["Cerebellar"]);
    expect(grouped["Cerebellar"].map((m) => m.name)).toEqual(["GolgiCell"]);
  });

  it("restricts the catalogue to a minimum evidence tier", () => {
    const curated = filterAndGroupModels(CATALOGUE, {
      modelFilter: "",
      familyFilter: "",
      patternFilter: "",
      minTier: 2,
      behaviors: NONE,
    });
    expect(Object.values(curated).flat().map((m) => m.name).sort()).toEqual([
      "AdExNeuron",
      "GLIFNeuron",
      "RulkovMapNeuron",
    ]);

    const verified = filterAndGroupModels(CATALOGUE, {
      modelFilter: "",
      familyFilter: "",
      patternFilter: "",
      minTier: 3,
      behaviors: NONE,
    });
    expect(Object.values(verified).flat().map((m) => m.name).sort()).toEqual([
      "GLIFNeuron",
      "RulkovMapNeuron",
    ]);
  });

  it("filters by search text across name and category", () => {
    const grouped = filterAndGroupModels(CATALOGUE, {
      modelFilter: "map",
      familyFilter: "",
      patternFilter: "",
      minTier: 0,
      behaviors: NONE,
    });
    expect(Object.keys(grouped)).toEqual(["Map-based"]);
  });
});
