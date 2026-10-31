// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Trace actions through the actual public Studio store
// @vitest-environment happy-dom

import { beforeEach, afterEach, expect, it, vi } from "vitest";
import { useStudioStore } from "./studio";
import type { SimulateResponse } from "../api/types";

const initial = useStudioStore.getState();
const values = Array.from({ length: 30 }, (_, index) => index);
const trace: SimulateResponse = {
  time: values.map((_, index) => (index + 1) * 5), states: { v: values },
  current_trace: values.map(() => 10), dt: 5, n_steps: 30,
  spikes: [3, 10, 20], spike_count: 3, model_name: "legacy-trace",
  stats: { isi_cv: null, isi_histogram: null, isi_mean_ms: null, rate_hz: 20 },
  run_metadata: { dt: 5, evidence_classification: "simulation", input_sha256: "a".repeat(64),
    n_steps: 30, result_sha256: "b".repeat(64), sample_count: 30,
    schema_version: "studio.simulation-run.v2", source: "model", spike_count: 3,
    status: "completed", state_variables: ["v"] },
};
const omitted: SimulateResponse = { ...trace, raw: {
  schema_version: "studio.raw-trace.v1", included: false,
  element_count: 2_100_000, element_budget: 2_000_000, dt: 5, n_steps: 100_000,
  sample_time_ms: "(index + 1) * dt", drive_interval_ms: "[index * dt, (index + 1) * dt)",
  spike_indices: trace.spikes, spike_times_ms: [20, 55, 105], vector_snapshots_only: [],
  reason: "Raw traces exceed the data limit.",
} };

beforeEach(() => { useStudioStore.setState({ ...initial, result: null, error: null, activeTab: "trace", staResult: null }, true); });
afterEach(() => { vi.restoreAllMocks(); useStudioStore.setState(initial, true); });

it("refuses unavailable CSV and STA through the public actions without making a download", () => {
  const urls = vi.spyOn(URL, "createObjectURL");
  useStudioStore.setState({ result: omitted });
  useStudioStore.getState().exportCSV();
  expect(useStudioStore.getState().error).toContain("CSV export requires full-resolution traces.");
  useStudioStore.getState().computeSTA();
  expect(useStudioStore.getState().error).toContain("Spike-triggered average requires full-resolution traces.");
  expect(useStudioStore.getState().activeTab).toBe("trace");
  expect(useStudioStore.getState().staResult).toBeNull();
  expect(urls).not.toHaveBeenCalled();
});

it("keeps legacy complete-trace downloads and averages working through their real browser sink", async () => {
  const urls = vi.spyOn(URL, "createObjectURL");
  useStudioStore.setState({ result: trace });
  useStudioStore.getState().exportCSV();
  useStudioStore.getState().exportData();
  useStudioStore.getState().exportSVG();
  expect(urls).toHaveBeenCalledTimes(3);
  const payloads: string[] = [];
  for (const [payload] of urls.mock.calls) {
    if (!(payload instanceof Blob)) throw new Error("A trace download must carry actual Blob bytes");
    payloads.push(await payload.text());
  }
  expect(payloads[0]?.split("\n")).toHaveLength(31);
  expect(payloads[1]).toContain('"n_steps": 30');
  expect(payloads[2]).toContain("<svg");
  useStudioStore.getState().computeSTA();
  expect(useStudioStore.getState().staResult).toEqual({ average: [9, 10, 11, 12], n_spikes: 3, time_ms: [-10, -5, 0, 5] });
  expect(useStudioStore.getState().activeTab).toBe("sta");
  expect(useStudioStore.getState().error).toBeNull();
});

it("retains the existing empty-result and insufficient-spike no-op actions", () => {
  useStudioStore.getState().exportData();
  useStudioStore.getState().exportCSV();
  useStudioStore.getState().exportSVG();
  useStudioStore.getState().computeSTA();
  useStudioStore.setState({ result: { ...trace, spikes: [3, 10], spike_count: 2 } });
  useStudioStore.getState().computeSTA();
  expect(useStudioStore.getState().staResult).toBeNull();
  expect(useStudioStore.getState().activeTab).toBe("trace");
  expect(useStudioStore.getState().error).toBeNull();
});
