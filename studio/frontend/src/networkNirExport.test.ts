// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio network NIR export helper tests

import { describe, expect, it } from "vitest";

import type { NIRFormat } from "./api/client";
import {
  NETWORK_NIR_EXPORT_FILENAME,
  networkNirExport,
  networkNirExportPlan,
  networkNirJson,
} from "./networkNirExport";

const nir: NIRFormat = {
  format: "nir",
  version: "1.0",
  nodes: {
    population: { type: "lif", size: 4 },
  },
  edges: [{ source: "population", target: "population" }],
};

describe("network NIR export", () => {
  it("uses the canonical Studio network NIR filename", () => {
    expect(NETWORK_NIR_EXPORT_FILENAME).toBe("network.nir.json");
  });

  it("serialises exported NIR with stable indentation", () => {
    expect(networkNirJson(nir)).toBe(JSON.stringify(nir, null, 2));
  });

  it("builds browser download artefacts with the canonical filename and MIME type", async () => {
    const exported = networkNirExport(nir);

    expect(exported.filename).toBe(NETWORK_NIR_EXPORT_FILENAME);
    expect(exported.blob.type).toBe("application/json");
    await expect(exported.blob.text()).resolves.toBe(JSON.stringify(nir, null, 2));
  });

  it("plans browser downloads with an injectable writer", () => {
    const plan = networkNirExportPlan(nir);
    const downloads: Array<{ filename: string; payload: Blob }> = [];

    plan.writeArtefact((payload, filename) => {
      downloads.push({ filename, payload });
    });

    expect(plan.artefact.filename).toBe(NETWORK_NIR_EXPORT_FILENAME);
    expect(downloads).toEqual([{
      filename: NETWORK_NIR_EXPORT_FILENAME,
      payload: plan.artefact.blob,
    }]);
  });
});
