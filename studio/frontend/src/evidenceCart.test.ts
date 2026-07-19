// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio evidence cart tests
import { describe, expect, it } from "vitest";

import {
  analysisCartDraft,
  buildEvidenceCartExport,
  clearEvidenceCart,
  emptyEvidenceCart,
  enqueueEvidenceCartArtefact,
  evidenceCartExportFilename,
  evidenceCartExportToBlob,
  evidenceCartHasSimAndAnalysis,
  removeEvidenceCartArtefact,
  sha256HexOfCanonicalJson,
  simulationCartDraft,
  EVIDENCE_CART_SCHEMA_VERSION,
} from "./evidenceCart";

describe("evidence cart queue", () => {
  it("starts empty with the cart schema version", () => {
    const cart = emptyEvidenceCart();
    expect(cart.items).toEqual([]);
    expect(cart.schema_version).toBe(EVIDENCE_CART_SCHEMA_VERSION);
  });

  it("queues simulation and analysis artefacts without mutating the prior cart", () => {
    const empty = emptyEvidenceCart();
    const sim = enqueueEvidenceCartArtefact(
      empty,
      simulationCartDraft("AdExNeuron", { spikes: [1, 2], I: 10 }, "simulation"),
      { id: "ec_sim", nowUtc: "2026-07-19T12:00:00.000Z" },
    );
    expect(sim.ok).toBe(true);
    if (!sim.ok) {
      return;
    }
    expect(empty.items).toHaveLength(0);
    expect(sim.cart.items).toHaveLength(1);
    expect(sim.item.kind).toBe("simulation");
    expect(sim.item.sourceName).toBe("AdExNeuron");

    const analysis = enqueueEvidenceCartArtefact(
      sim.cart,
      analysisCartDraft("AdExNeuron", { f_I: [{ I: 10, rate: 12.5 }] }, "analysis"),
      { id: "ec_an", nowUtc: "2026-07-19T12:00:01.000Z" },
    );
    expect(analysis.ok).toBe(true);
    if (!analysis.ok) {
      return;
    }
    expect(sim.cart.items).toHaveLength(1);
    expect(analysis.cart.items).toHaveLength(2);
    expect(evidenceCartHasSimAndAnalysis(analysis.cart)).toBe(true);
    expect(evidenceCartHasSimAndAnalysis(sim.cart)).toBe(false);
  });

  it("rejects empty labels and non-serialisable payloads", () => {
    const cart = emptyEvidenceCart();
    const blank = enqueueEvidenceCartArtefact(cart, {
      classification: "simulation",
      kind: "simulation",
      label: "   ",
      payload: { ok: true },
    });
    expect(blank.ok).toBe(false);
    if (blank.ok) {
      return;
    }
    expect(blank.error).toMatch(/label/i);

    const bad = enqueueEvidenceCartArtefact(cart, {
      classification: "simulation",
      kind: "simulation",
      label: "broken",
      payload: { n: Number.NaN },
    });
    expect(bad.ok).toBe(false);
  });

  it("removes items by id and clears the cart", () => {
    let cart = emptyEvidenceCart();
    const first = enqueueEvidenceCartArtefact(
      cart,
      simulationCartDraft("m", { a: 1 }),
      { id: "ec_1" },
    );
    expect(first.ok).toBe(true);
    if (!first.ok) {
      return;
    }
    cart = first.cart;
    const second = enqueueEvidenceCartArtefact(
      cart,
      analysisCartDraft("m", { b: 2 }),
      { id: "ec_2" },
    );
    expect(second.ok).toBe(true);
    if (!second.ok) {
      return;
    }
    cart = second.cart;
    cart = removeEvidenceCartArtefact(cart, "ec_1");
    expect(cart.items.map((item) => item.id)).toEqual(["ec_2"]);
    const unchanged = removeEvidenceCartArtefact(cart, "missing");
    expect(unchanged).toBe(cart);
    expect(clearEvidenceCart().items).toEqual([]);
  });
});

describe("evidence cart export digests", () => {
  it("builds a single export with input-dependent digests for multiple artefacts", async () => {
    let cart = emptyEvidenceCart();
    const simPayload = { model: "AdExNeuron", duration: 100, spikes: [10, 20, 30] };
    const analysisPayload = { model: "AdExNeuron", metric: "f_I", points: 5 };
    const sim = enqueueEvidenceCartArtefact(
      cart,
      simulationCartDraft("AdExNeuron", simPayload),
      { id: "ec_sim", nowUtc: "2026-07-19T12:00:00.000Z" },
    );
    expect(sim.ok).toBe(true);
    if (!sim.ok) {
      return;
    }
    cart = sim.cart;
    const analysis = enqueueEvidenceCartArtefact(
      cart,
      analysisCartDraft("AdExNeuron", analysisPayload),
      { id: "ec_an", nowUtc: "2026-07-19T12:00:01.000Z" },
    );
    expect(analysis.ok).toBe(true);
    if (!analysis.ok) {
      return;
    }
    cart = analysis.cart;

    const exportA = await buildEvidenceCartExport(cart, {
      exportedAtUtc: "2026-07-19T12:05:00.000Z",
    });
    expect("error" in exportA).toBe(false);
    if ("error" in exportA) {
      return;
    }
    expect(exportA.schema_version).toBe(EVIDENCE_CART_SCHEMA_VERSION);
    expect(exportA.entry_count).toBe(2);
    expect(exportA.entries).toHaveLength(2);
    expect(exportA.kind_counts).toEqual({ analysis: 1, simulation: 1 });
    expect(exportA.bundle_sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(exportA.entries[0]?.payload_sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(exportA.entries[1]?.payload_sha256).toMatch(/^[0-9a-f]{64}$/);

    // Digests must follow the real payloads (no hardcoded goldens that ignore inputs).
    const expectedSimDigest = await sha256HexOfCanonicalJson(simPayload);
    const expectedAnalysisDigest = await sha256HexOfCanonicalJson(analysisPayload);
    expect(exportA.entries[0]?.payload_sha256).toBe(expectedSimDigest);
    expect(exportA.entries[1]?.payload_sha256).toBe(expectedAnalysisDigest);
    expect(exportA.entries[0]?.payload).toEqual(simPayload);
    expect(exportA.entries[1]?.payload).toEqual(analysisPayload);

    // Key order must not change digests (canonical JSON).
    const reordered = await sha256HexOfCanonicalJson({
      spikes: [10, 20, 30],
      duration: 100,
      model: "AdExNeuron",
    });
    expect(reordered).toBe(expectedSimDigest);

    // Different inputs must change digests.
    const otherSim = await sha256HexOfCanonicalJson({
      ...simPayload,
      spikes: [10, 20, 31],
    });
    expect(otherSim).not.toBe(expectedSimDigest);

    // Bundle digest is stable for identical cart + timestamp.
    const exportB = await buildEvidenceCartExport(cart, {
      exportedAtUtc: "2026-07-19T12:05:00.000Z",
    });
    expect("error" in exportB).toBe(false);
    if ("error" in exportB) {
      return;
    }
    expect(exportB.bundle_sha256).toBe(exportA.bundle_sha256);

    const blob = evidenceCartExportToBlob(exportA);
    expect(blob.type).toBe("application/json");
    expect(blob.size).toBeGreaterThan(32);
    expect(evidenceCartExportFilename(exportA)).toBe(
      `studio-evidence-cart-${exportA.bundle_sha256.slice(0, 12)}.json`,
    );
  });

  it("refuses to export an empty cart", async () => {
    const result = await buildEvidenceCartExport(emptyEvidenceCart());
    expect(result).toEqual({
      error: "Evidence cart is empty; queue at least one artefact",
    });
  });
});
