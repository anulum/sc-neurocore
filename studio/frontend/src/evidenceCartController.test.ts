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

/**
 * Build a simulation response, overridden field by field.
 *
 * @param overrides - The fields to change.
 * @returns The response.
 */
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
  it.each([true, false])("round-trips complete simulation evidence, raw included=%s", async (included) => {
    const result = simulateResult({
      spikes: [2], time: [0.1, 0.2, 0.3],
      initial_state: { v: -65, synapses: [0, 1] },
      final_state: { v: -55, synapses: [2, 3] },
      raw: { schema_version: "studio.raw-trace.v1", included,
        element_count: 12, element_budget: included ? 100 : 0, dt: 0.1, n_steps: 3,
        sample_time_ms: "(index + 1) * dt", drive_interval_ms: "[index * dt, (index + 1) * dt)",
        spike_indices: [2], spike_times_ms: [0.3], vector_snapshots_only: [],
        ...(included ? { states: { v: [-65, -60, -55] }, vector_states: { synapses: [[0, 1], [1, 2], [2, 3]] }, drive: [10, 10, 10] }
          : { reason: "raw element budget exceeded" }) },
      display: { schema_version: "studio.display-projection.v1", method: "identity", max_points: 3,
        bucket_count: 0, point_count: 3, sample_index: [0, 1, 2], first_sample_included: true,
        final_sample_included: true, spikes_are_raw_steps: true },
      experiment: { schema_version: "studio.experiment-spec.v1", source: "model", model: { name: "LIFNeuron" },
        numerical: { method: "euler", family: "ode", dt: 0.1, dt_source: "request", substeps: 1, time_unit: "ms" },
        steps: { n_steps: 3, duration_requested_ms: 0.3, duration_effective_ms: 0.3, synchronous_limit: 100 },
        parameters: { tau: 10 }, initial_state: { v: -65 }, initial_state_source: "model-default",
        protocol: { kind: "constant", current: 10 },
        randomness: { kind: "none", seed: null, seed_source: "none", trial: "replay", effective_trial: "replay", generator: null },
        backend: { selected: "python", rejected: [] }, runtime: { python: "controlled" },
        experiment_sha256: "c".repeat(64), cache: { key: "controlled", cacheable: true } },
    });
    const queued = decideSimulationEnqueue(emptyEvidenceCart(), { runSucceeded: true,
      sourceMode: "model", selectedModelName: "LIFNeuron", result, resultIdentityBefore: null });
    expect(queued.action).toBe("enqueue");
    if (queued.action !== "enqueue") throw new Error("Simulation was not queued");
    const exported = await exportEvidenceCartWithVerification(queued.cart);
    expect(exported.ok).toBe(true);
    if (!exported.ok) throw new Error(exported.error);
    const wire: unknown = JSON.parse(await exported.blob.text());
    expect(wire).toMatchObject({ entries: [{ payload: { ...result, source_mode: "model" } }] });
    expect(exported.bundle.entries[0]?.payload).toEqual({ ...result, source_mode: "model" });
  });
  it("includes states and ODE identity; skips failed or unchanged results", () => {
    const priorMeta = {
      dt: 0.1,
      evidence_classification: "simulation" as const,
      input_sha256: "a".repeat(64),
      n_steps: 3,
      result_sha256: "0".repeat(64),
      sample_count: 3,
      schema_version: "studio.simulation-run.v1" as const,
      source: "model" as const,
      spike_count: 0,
      status: "completed" as const,
      state_variables: ["v"],
    };
    const successMeta = {
      ...priorMeta,
      result_sha256: "b".repeat(64),
      spike_count: 1,
    };
    const before = simulationResultIdentity(
      simulateResult({ spike_count: 0, run_metadata: priorMeta }),
    );
    const success = decideSimulationEnqueue(emptyEvidenceCart(), {
      runSucceeded: true,
      sourceMode: "ode",
      selectedModelName: "stale-model",
      result: simulateResult({ model_name: undefined, run_metadata: successMeta }),
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

  it("enqueues same-shape results when result_sha256 differs (no shape-only collision)", () => {
    const shape = {
      time: [0, 0.1, 0.2],
      states: { v: [-65, -60, -55] as number[] },
      current_trace: [10, 10, 10],
      spikes: [0.2],
      spike_count: 1,
      dt: 0.1,
      n_steps: 3,
    };
    const first = simulateResult({
      ...shape,
      run_metadata: {
        dt: 0.1,
        evidence_classification: "simulation",
        input_sha256: "a".repeat(64),
        n_steps: 3,
        result_sha256: "1".repeat(64),
        sample_count: 3,
        schema_version: "studio.simulation-run.v1",
        source: "model",
        spike_count: 1,
        status: "completed",
        state_variables: ["v"],
      },
    });
    const second = simulateResult({
      ...shape,
      // Distinct values, identical counts/lengths/keys/dt.
      states: { v: [-64, -59, -50] },
      spikes: [0.15],
      run_metadata: {
        dt: 0.1,
        evidence_classification: "simulation",
        input_sha256: "c".repeat(64),
        n_steps: 3,
        result_sha256: "2".repeat(64),
        sample_count: 3,
        schema_version: "studio.simulation-run.v1",
        source: "model",
        spike_count: 1,
        status: "completed",
        state_variables: ["v"],
      },
    });
    expect(simulationResultIdentity(first)).not.toBe(simulationResultIdentity(second));
    expect(first.spike_count).toBe(second.spike_count);
    expect(first.n_steps).toBe(second.n_steps);
    expect(first.dt).toBe(second.dt);
    expect(Object.keys(first.states)).toEqual(Object.keys(second.states));

    const afterFirst = decideSimulationEnqueue(emptyEvidenceCart(), {
      runSucceeded: true,
      sourceMode: "model",
      selectedModelName: "LIFNeuron",
      result: first,
      resultIdentityBefore: null,
    });
    expect(afterFirst.action).toBe("enqueue");
    if (afterFirst.action !== "enqueue") {
      return;
    }
    const afterSecond = decideSimulationEnqueue(afterFirst.cart, {
      runSucceeded: true,
      sourceMode: "model",
      selectedModelName: "LIFNeuron",
      result: second,
      resultIdentityBefore: simulationResultIdentity(first),
    });
    expect(afterSecond.action).toBe("enqueue");
    if (afterSecond.action !== "enqueue") {
      return;
    }
    expect(afterSecond.cart.items).toHaveLength(2);
    expect((afterSecond.cart.items[1]?.payload as { states: { v: number[] } }).states.v)
      .toEqual([-64, -59, -50]);
  });
});

const ANALYSIS_DIGEST_A = "a".repeat(64);
const ANALYSIS_DIGEST_B = "b".repeat(64);
const ANALYSIS_KINDS = [
  "fi_curve",
  "bifurcation",
  "sensitivity",
  "heatmap",
  "other",
] as const;

describe("decideAnalysisEnqueue", () => {
  it("queues only the exact analysis result and does not bag unrelated fields", () => {
    const decision = decideAnalysisEnqueue(emptyEvidenceCart(), {
      runSucceeded: true,
      sourceMode: "model",
      selectedModelName: "AdExNeuron",
      analysisKind: "fi_curve",
      analysisResult: { currents: [0, 1], rates: [0, 5] },
      resultIdentityBefore: null,
      resultIdentityAfter: ANALYSIS_DIGEST_A,
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
    if (decision.action === "skip") {
      expect(decision.reason).toBe("analysis_run_failed");
    }
  });

  it("skips unchanged digests for every analysis kind", () => {
    for (const analysisKind of ANALYSIS_KINDS) {
      const cart = emptyEvidenceCart();
      const decision = decideAnalysisEnqueue(cart, {
        runSucceeded: true,
        sourceMode: "model",
        selectedModelName: "AdExNeuron",
        analysisKind,
        analysisResult: { kind: analysisKind, value: 1 },
        resultIdentityBefore: ANALYSIS_DIGEST_A,
        resultIdentityAfter: ANALYSIS_DIGEST_A,
      });
      expect(decision.action).toBe("skip");
      if (decision.action === "skip") {
        expect(decision.reason).toBe("analysis_result_unchanged");
        expect(decision.cart).toBe(cart);
        expect(decision.cart.items).toHaveLength(0);
      }
    }
  });

  it("enqueues when digest changes even if payload shape matches", () => {
    const decision = decideAnalysisEnqueue(emptyEvidenceCart(), {
      runSucceeded: true,
      sourceMode: "ode",
      selectedModelName: "",
      analysisKind: "heatmap",
      analysisResult: { rates: [[1]], shape: "same" },
      resultIdentityBefore: ANALYSIS_DIGEST_A,
      resultIdentityAfter: ANALYSIS_DIGEST_B,
    });
    expect(decision.action).toBe("enqueue");
    if (decision.action !== "enqueue") {
      return;
    }
    expect(decision.cart.items).toHaveLength(1);
    expect(
      (decision.cart.items[0]?.payload as { analysis_kind: string }).analysis_kind,
    ).toBe("heatmap");
  });

  it("fails closed when a claimed success lacks a valid after identity", () => {
    const cart = emptyEvidenceCart();
    const decision = decideAnalysisEnqueue(cart, {
      runSucceeded: true,
      sourceMode: "model",
      selectedModelName: "AdExNeuron",
      analysisKind: "fi_curve",
      analysisResult: { currents: [0], rates: [1] },
      resultIdentityBefore: ANALYSIS_DIGEST_A,
      resultIdentityAfter: null,
    });
    expect(decision.action).toBe("skip");
    if (decision.action === "skip") {
      expect(decision.reason).toBe("analysis_result_identity_invalid");
      expect(decision.cart).toBe(cart);
      expect(decision.cart.items).toHaveLength(0);
    }
  });
});

describe("export freshness and payload round-trip", () => {
  it.each(["payload", "label", "identity"] as const)("rejects same-count cart changes: %s", async (change) => {
    const queued = enqueueEvidenceCartArtefact(emptyEvidenceCart(), simulationCartDraft("m", { values: [1, 2] }));
    if (!queued.ok) throw new Error(queued.error);
    const exported = await exportEvidenceCartWithVerification(queued.cart);
    if (!exported.ok) throw new Error(exported.error);
    const changed = { ...queued.cart, items: queued.cart.items.map((item) => ({ ...item,
      ...(change === "payload" ? { payload: { values: [1, 3] } }
        : change === "label" ? { label: "Changed label" } : { id: "ec_replacement" }),
    })) };
    expect(evidenceCartExportSatisfiesGuided(changed, exported.bundle, 1)).toBe(false);
    expect(evidenceCartExportSatisfiesGuided(queued.cart, exported.bundle, 1)).toBe(true);
  });

  it("snapshots payloads before asynchronous export hashing", async () => {
    const payload = { values: [1, 2] };
    const queued = enqueueEvidenceCartArtefact(emptyEvidenceCart(), simulationCartDraft("m", payload));
    if (!queued.ok) throw new Error(queued.error);
    const pending = exportEvidenceCartWithVerification(queued.cart);
    payload.values[1] = 3;
    const exported = await pending;
    expect(exported.ok).toBe(true);
    if (!exported.ok) throw new Error(exported.error);
    expect(exported.bundle.entries[0]?.payload).toEqual({ values: [1, 2] });
    expect(evidenceCartExportSatisfiesGuided(queued.cart, exported.bundle, 1)).toBe(false);
  });
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
