// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — portable event input recovery acceptance

import { describe, expect, it } from "vitest";
import eventData from "./studioEventTrainingData.fixture.json";
import { decodeTrainingJobSummaries, observedTrainingConfig } from "./studioTrainingRecovery";

const config = {
  schema_version: "studio.training-config.v1", dataset: "nmnist", epochs: 1,
  batch_size: 3, lr: 0.001, hidden: [4], timesteps: 4,
  surrogate: "atan_surrogate", learn_beta: false, learn_threshold: false,
  max_grad_norm: 0, seed: 7, event_data: eventData,
};

/**
 * Decode an actual retained-list envelope through its public consumer.
 *
 * @param event - Portable event field, including intentionally malformed inputs.
 * @returns The recovered editable provenance.
 */
function recover(event: unknown) {
  const jobs = decodeTrainingJobSummaries([{
    job_id: "sj_event", status: "completed", config: { ...config, event_data: event },
  }]);
  return observedTrainingConfig(jobs[0]?.config ?? null);
}

/**
 * Copy a producer declaration for a single incompatible envelope change.
 *
 * @returns A JSON copy with object fields safe to alter independently.
 */
function declaration(): Record<string, unknown> {
  return structuredClone(eventData);
}

describe("portable event input at retained-job recovery", () => {
  it("preserves every producer declaration without rewriting its digests or paths", () => {
    expect(recover(eventData)).toMatchObject({ event_data: eventData });
  });

  it.each([
    null, [], { ...eventData, schema: "future" },
    { ...eventData, training_cpu_threads: 2 },
    { ...eventData, train_split: "" },
    { ...eventData, evaluation_split: eventData.train_split },
    { ...eventData, extra: "ignored" },
    { ...eventData, encoder: { ...eventData.encoder, n_steps: 5 } },
    { ...eventData, digests: { ...eventData.digests, encoder: "sha256:invalid" } },
    { ...eventData, digests: { ...eventData.digests, extra: "ignored" } },
    { ...eventData, split: { ...eventData.split, manifest_digest: "sha256:" + "0".repeat(64) } },
  ])("refuses an incompatible retained input rather than displaying partial provenance", (event) => {
    expect(() => recover(event)).toThrow("invalid configuration");
  });

  it("refuses a declaration whose required envelope field was removed", () => {
    const event = declaration();
    delete event.training_cpu_threads;
    expect(() => recover(event)).toThrow("invalid configuration");
  });

  it("refuses an event input attached to a static run", () => {
    expect(() => decodeTrainingJobSummaries([{
      job_id: "sj_static", status: "completed", config: { ...config, dataset: "mnist" },
    }])).toThrow("invalid configuration");
  });
});
