// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio network NIR export and import helper tests

import { describe, expect, it } from "vitest";

import type { GraphEnvelope, NIRExportResult, NIRImportResult } from "./api/client";
import {
  NETWORK_NIR_EXPORT_FILENAME,
  NETWORK_NIR_MEDIA_TYPE,
  base64Bytes,
  bytesBase64,
  networkNirExport,
  networkNirExportNotice,
  networkNirExportPlan,
  networkNirImportNotice,
  networkNirImportRequest,
} from "./networkNirExport";

/** The eight-byte signature every HDF5 file, and so every NIR file, begins with. */
const HDF5_SIGNATURE = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];

const exported: NIRExportResult = {
  schema_version: "sc-neurocore.studio.nir-export.v1",
  filename: "network.nir",
  media_type: "application/x-hdf5",
  nir_version: "1.0.7",
  content_base64: bytesBase64(new Uint8Array([...HDF5_SIGNATURE, 0, 255, 7])),
  notes: ["times are in milliseconds", "population a: fires at v >= v_threshold"],
};

describe("network NIR export", () => {
  it("names the file what it is: NIR in HDF5", () => {
    expect(NETWORK_NIR_EXPORT_FILENAME).toBe("network.nir");
    expect(NETWORK_NIR_MEDIA_TYPE).toBe("application/x-hdf5");
  });

  it("writes the file's bytes, not the JSON that carried them", async () => {
    const artefact = networkNirExport(exported);

    expect(artefact.filename).toBe(NETWORK_NIR_EXPORT_FILENAME);
    expect(artefact.blob.type).toBe(NETWORK_NIR_MEDIA_TYPE);
    const bytes = new Uint8Array(await artefact.blob.arrayBuffer());
    expect([...bytes]).toEqual([...HDF5_SIGNATURE, 0, 255, 7]);
  });

  it("round-trips every byte value through base64, across chunk boundaries", () => {
    const bytes = new Uint8Array(0x8000 * 2 + 3);
    for (let index = 0; index < bytes.length; index += 1) bytes[index] = index % 256;

    expect(base64Bytes(bytesBase64(bytes))).toEqual(bytes);
  });

  it("plans browser downloads with an injectable writer", () => {
    const plan = networkNirExportPlan(exported);
    const downloads: { filename: string; payload: Blob }[] = [];

    plan.writeArtefact((payload, filename) => {
      downloads.push({ filename, payload });
    });

    expect(downloads).toEqual([{ filename: NETWORK_NIR_EXPORT_FILENAME, payload: plan.artefact.blob }]);
  });

  it("says what the file does not carry", () => {
    expect(networkNirExportNotice(exported)).toBe(
      "Exported network.nir (NIR 1.0.7). Not carried exactly: times are in milliseconds; "
        + "population a: fires at v >= v_threshold.",
    );
  });
});

describe("network NIR import", () => {
  it("sends a NIR file as the base64 of its bytes", async () => {
    const file = new File([new Uint8Array([...HDF5_SIGNATURE, 1, 2])], "model.nir");

    const request = await networkNirImportRequest(file);

    expect(request).toEqual({ content_base64: bytesBase64(new Uint8Array([...HDF5_SIGNATURE, 1, 2])) });
  });

  it("sends a saved .json envelope as the envelope it is", async () => {
    const envelope: GraphEnvelope = { format: "sc-neurocore.studio.graph", version: "2", nodes: {}, edges: [] };
    const file = new File([JSON.stringify(envelope)], "Network.JSON");

    expect(await networkNirImportRequest(file)).toEqual(envelope);
  });

  it.each([
    ["studio", [], "Imported a NIR file this Studio wrote; every tensor matched its recorded network."],
    ["foreign", ["a: NIR LIF fires at v > v_threshold"], "Imported a NIR file from another tool. Assumed: a: NIR LIF fires at v > v_threshold."],
    ["studio-envelope", ["legacy envelope"], "Imported a saved graph envelope. Assumed: legacy envelope."],
  ] as const)("names a %s import and what it assumed", (origin, notes, notice) => {
    const imported: NIRImportResult = {
      graph: { populations: [], projections: [] },
      origin,
      notes: [...notes],
    };

    expect(networkNirImportNotice(imported)).toBe(notice);
  });
});
