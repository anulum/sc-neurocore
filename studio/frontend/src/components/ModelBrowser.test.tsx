// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { describe, expect, it } from "vitest";

import type { ModelBehavior, ModelScanMetadata } from "../api/client";
import { buildModelScanEvidenceItems, filterAndGroupModels } from "./ModelBrowser";

interface BrowseModel {
  name: string;
  category: string;
  family: string;
  tier: number;
  science_tier: number;
  silicon_tier: number | null;
}

const CATALOGUE: BrowseModel[] = [
  {
    name: "AdExNeuron",
    category: "Integrate-and-Fire",
    family: "Integrate-and-Fire",
    tier: 2,
    science_tier: 5,
    silicon_tier: 1,
  },
  {
    name: "GLIFNeuron",
    category: "Integrate-and-Fire",
    family: "Integrate-and-Fire",
    tier: 3,
    science_tier: 3,
    silicon_tier: 0,
  },
  {
    name: "GolgiCell",
    category: "Cerebellar",
    family: "Cerebellar",
    tier: 1,
    science_tier: 1,
    silicon_tier: null,
  },
  {
    name: "RulkovMapNeuron",
    category: "Map-based",
    family: "Map-based",
    tier: 3,
    science_tier: 5,
    silicon_tier: 2,
  },
];

const NONE: Record<string, ModelBehavior> = {};

const scanMetadata: ModelScanMetadata = {
  current: 10,
  duration: 100,
  error_count: 2,
  evidence_classification: "analysis",
  failed_models: [
    { name: "DendriticNMDANeuron", category: "Synaptic", error_type: "TypeError", error_message: "needs glutamate" },
    { name: "ChayKeizerNeuron", category: "Bursting", error_type: "ValueError", error_message: "outside safety envelope" },
  ],
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
      { label: "errors", value: "2" },
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

  it("restricts the catalogue to a measured behaviour tag", () => {
    const tagged = [
      { ...CATALOGUE[0], behavior_tags: ["excitable", "tonic", "rate-coded"] },
      { ...CATALOGUE[1], behavior_tags: ["excitable", "adapting"] },
      { ...CATALOGUE[2], behavior_tags: ["quiescent"] },
      { ...CATALOGUE[3], behavior_tags: ["excitable", "bursting"] },
    ];
    const adapting = filterAndGroupModels(tagged, {
      modelFilter: "",
      familyFilter: "",
      patternFilter: "",
      minTier: 0,
      behaviors: NONE,
      behaviorFilter: "adapting",
    });
    expect(Object.values(adapting).flat().map((m) => m.name)).toEqual(["GLIFNeuron"]);

    const excitable = filterAndGroupModels(tagged, {
      modelFilter: "",
      familyFilter: "",
      patternFilter: "",
      minTier: 0,
      behaviors: NONE,
      behaviorFilter: "excitable",
    });
    expect(Object.values(excitable).flat().map((m) => m.name).sort()).toEqual([
      "AdExNeuron",
      "GLIFNeuron",
      "RulkovMapNeuron",
    ]);
  });

  it("ignores a behaviour filter that no model carries", () => {
    const tagged = [{ ...CATALOGUE[0], behavior_tags: ["excitable", "tonic"] }];
    const grouped = filterAndGroupModels(tagged, {
      modelFilter: "",
      familyFilter: "",
      patternFilter: "",
      minTier: 0,
      behaviors: NONE,
      behaviorFilter: "chaotic",
    });
    expect(Object.values(grouped).flat()).toEqual([]);
  });

  it("restricts the catalogue to a minimum science dual-axis tier", () => {
    const s5 = filterAndGroupModels(CATALOGUE, {
      modelFilter: "",
      familyFilter: "",
      patternFilter: "",
      minTier: 0,
      minScienceTier: 5,
      behaviors: NONE,
    });
    expect(Object.values(s5).flat().map((m) => m.name).sort()).toEqual([
      "AdExNeuron",
      "RulkovMapNeuron",
    ]);

    const s3 = filterAndGroupModels(CATALOGUE, {
      modelFilter: "",
      familyFilter: "",
      patternFilter: "",
      minTier: 0,
      minScienceTier: 3,
      behaviors: NONE,
    });
    expect(Object.values(s3).flat().map((m) => m.name).sort()).toEqual([
      "AdExNeuron",
      "GLIFNeuron",
      "RulkovMapNeuron",
    ]);
  });

  it("restricts the catalogue to silicon-enrolled models and H floors", () => {
    const enrolled = filterAndGroupModels(CATALOGUE, {
      modelFilter: "",
      familyFilter: "",
      patternFilter: "",
      minTier: 0,
      siliconEnrolledOnly: true,
      behaviors: NONE,
    });
    expect(Object.values(enrolled).flat().map((m) => m.name).sort()).toEqual([
      "AdExNeuron",
      "GLIFNeuron",
      "RulkovMapNeuron",
    ]);

    const h1 = filterAndGroupModels(CATALOGUE, {
      modelFilter: "",
      familyFilter: "",
      patternFilter: "",
      minTier: 0,
      minSiliconTier: 1,
      behaviors: NONE,
    });
    expect(Object.values(h1).flat().map((m) => m.name).sort()).toEqual([
      "AdExNeuron",
      "RulkovMapNeuron",
    ]);
  });

  it("combines dual-axis floors with family search", () => {
    const grouped = filterAndGroupModels(CATALOGUE, {
      modelFilter: "integrate",
      familyFilter: "",
      patternFilter: "",
      minTier: 0,
      minScienceTier: 5,
      minSiliconTier: 1,
      behaviors: NONE,
    });
    expect(Object.values(grouped).flat().map((m) => m.name)).toEqual(["AdExNeuron"]);
  });
});
