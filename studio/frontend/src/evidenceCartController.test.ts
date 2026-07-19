// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Evidence cart controller tests
import { describe, expect, it } from "vitest";

import type { SimulateResponse } from "./api/client";
import {
  emptyEvidenceCart,
  enqueueEvidenceCartArtefact,
  simulationCartDraft,
  verifyEvidenceCartExportRoundTrip,
  buildEvidenceCartExport,
} from "./evidenceCart";
import {
  decideAnalysisEnqueue,
  decideSimulationEnqueue,
  evidenceCartExportSatisfiesGuided,
  exportEvidenceCartWithVerification,
  simulationResultIdentity,
} from "./evidenceCartController";

function simulateResult(overrides: Partial<SimulateResponse> = {}): SimulateResponse {
  return {
    time: [0, 0.1, 0.2],
    states: { v: [-65, -60, -55] },
    current_trace: [10, 10, 10],
    spikes: [0.2],
    spike_count: 1,
    stats: {
      rate_hz: 5,
      isi_mean_ms: null,
      isi_cv: null,
      isi_histogram: null,
    },
    dt: 0.1,
    n_steps: 3,
    model_name: "LIFNeuron",
    run_metadata: {
      dt: 0.1,
      evidence_classification: "simulation",
      input_sha256: "a".repeat(64),
      n_steps: 3,
      result_sha256: "b".repeat(64),
      sample_count: 3,
      schema_version: "studio.simulation-run.v1",
      source: "model",
      spike_count: 1,
      status: "completed",
      state_variables: ["v"],
    },
    ...overrides,
  };
}

describe("decideSimulationEnqueue", () => {
  it("includes states and ODE identity; skips failed or unchanged results", () => {
    const before = simulationResultIdentity(simulateResult({ spike_count: 0 }));
    const success = decideSimulationEnqueue(emptyEvidenceCart(), {
      runSucceeded: true,
      sourceMode: "ode",
      selectedModelName: "stale-model",
      result: simulateResult({ model_name: undefined }),
      resultIdentityBefore: before,
    });
    expect(success.action).toBe("enqueue");
    if (success.action !== "enqueue") {
      return;
    }
    const item = success.cart.items[0];
    expect(item?.sourceName).toBe("ode");
    expect((item?.payload as { states: unknown }).states).toEqual({ v: [-65, -60, -55] });
    expect((item?.payload as { source_mode: string }).source_mode).toBe("ode");

    const failed = decideSimulationEnqueue(emptyEvidenceCart(), {
      runSucceeded: false,
      sourceMode: "model",
      selectedModelName: "LIFNeuron",
      result: simulateResult(),
      resultIdentityBefore: null,
    });
    expect(failed.action).toBe("skip");

    const same = decideSimulationEnqueue(emptyEvidenceCart(), {
      runSucceeded: true,
      sourceMode: "model",
      selectedModelName: "LIFNeuron",
      result: simulateResult(),
      resultIdentityBefore: simulationResultIdentity(simulateResult()),
    });
    expect(same.action).toBe("skip");
    expect(same.action === "skip" && same.reason).toBe("simulation_result_unchanged");
  });
});

describe("decideAnalysisEnqueue", () => {
  it("queues only the exact analysis result and does not bag unrelated fields", () => {
    const decision = decideAnalysisEnqueue(emptyEvidenceCart(), {
      runSucceeded: true,
      sourceMode: "model",
      selectedModelName: "AdExNeuron",
      analysisKind: "fi_curve",
      analysisResult: { currents: [0, 1], rates: [0, 5] },
      resultIdentityBefore: null,
      resultIdentityAfter: "x",
    });
    expect(decision.action).toBe("enqueue");
    if (decision.action !== "enqueue") {
      return;
    }
    const payload = decision.cart.items[0]?.payload as {
      analysis_kind: string;
      result: { currents: number[] };
    };
    expect(payload.analysis_kind).toBe("fi_curve");
    expect(payload.result.currents).toEqual([0, 1]);
    expect(payload).not.toHaveProperty("bif");
  });

  it("skips failed analysis runs", () => {
    const decision = decideAnalysisEnqueue(emptyEvidenceCart(), {
      runSucceeded: false,
      sourceMode: "model",
      selectedModelName: "AdExNeuron",
      analysisKind: "fi_curve",
      analysisResult: { stale: true },
      resultIdentityBefore: null,
      resultIdentityAfter: null,
    });
    expect(decision.action).toBe("skip");
  });
});

describe("export freshness and payload round-trip", () => {
  it("invalidates guided export satisfaction after new cart items", async () => {
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
    const exported = await buildEvidenceCartExport(cart, {
      exportedAtUtc: "2026-07-19T12:00:00.000Z",
    });
    expect("error" in exported).toBe(false);
    if ("error" in exported) {
      return;
    }
    expect(
      evidenceCartExportSatisfiesGuided(cart, exported, cart.items.length),
    ).toBe(true);
    const second = enqueueEvidenceCartArtefact(
      cart,
      simulationCartDraft("m", { a: 2 }),
      { id: "ec_2" },
    );
    expect(second.ok).toBe(true);
    if (!second.ok) {
      return;
    }
    expect(
      evidenceCartExportSatisfiesGuided(second.cart, exported, cart.items.length),
    ).toBe(false);
  });

  it("exports payloads and verifies digest round-trip", async () => {
    const queued = enqueueEvidenceCartArtefact(
      emptyEvidenceCart(),
      simulationCartDraft("LIF", { states: { v: [1, 2] }, spikes: [1] }),
      { id: "ec_sim" },
    );
    expect(queued.ok).toBe(true);
    if (!queued.ok) {
      return;
    }
    const result = await exportEvidenceCartWithVerification(queued.cart, {
      exportedAtUtc: "2026-07-19T12:00:00.000Z",
    });
    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.bundle.entries[0]?.payload).toEqual({
      states: { v: [1, 2] },
      spikes: [1],
    });
    const verified = await verifyEvidenceCartExportRoundTrip(result.bundle);
    expect(verified).toEqual({ ok: true });
  });
});
