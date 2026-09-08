// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Real store evidence cart attribution
// @vitest-environment happy-dom

import { act, StrictMode, useEffect } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, expect, it, vi } from "vitest";
import { useEvidenceCartSession, type EvidenceCartSession } from "./useEvidenceCartSession";
import { useStudioStore } from "./stores/studio";

const initial = useStudioStore.getState();
let root: Root | undefined;
let unsubscribe: (() => void) | undefined;
afterEach(async () => {
  unsubscribe?.();
  await act(async () => { root?.unmount(); await Promise.resolve(); });
  useStudioStore.setState(initial, true);
  document.body.replaceChildren(); vi.unstubAllGlobals();
});

it.each((["simulation", "analysis"] as const).flatMap((kind) =>
  [false, true].map((change) => ({ kind, change }))))("attributes $kind to submitted source, changed after result=$change", async ({ kind, change }) => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  useStudioStore.setState({ sourceMode: "model", selectedModelName: "SCLapicqueLIFNeuron" });
  const runMetadata = { schema_version: "studio.simulation-run.v1", source: "model", status: "completed", evidence_classification: "simulation",
    input_sha256: "a".repeat(64), result_sha256: "b".repeat(64), dt: 0.1, n_steps: 1, sample_count: 1, spike_count: 0, state_variables: ["v"] };
  const result = kind === "simulation" ? { time: [0.1], states: { v: [-65] }, current_trace: [10], spikes: [], spike_count: 0,
    stats: { rate_hz: 0 }, dt: 0.1, n_steps: 1, model_name: "SCLapicqueLIFNeuron", run_metadata: runMetadata }
    : { currents: [0, 1], rates: [0, 5], analysis_metadata: { schema_version: "studio.analysis-result.v1", analysis_type: "fi_curve",
      source: "model", status: "completed", evidence_classification: "analysis", input_sha256: "a".repeat(64), result_sha256: "c".repeat(64), output_keys: ["currents", "rates"] } };
  const job = { job_id: "sj_cart", kind: "analysis", owner: "studio", status: "completed", result,
    artifacts: [], error: null, created_at_utc: "2026-09-08T00:00:00Z", started_at_utc: null, finished_at_utc: null, execution_model: "thread", request_id: null };
  vi.stubGlobal("fetch", vi.fn<typeof globalThis.fetch>(async (url) => new Response(JSON.stringify(
    kind === "simulation" ? result : url === "/api/studio/jobs/sj_cart" ? job : {
      analysis: "fi_curve", execution_mode: "async_job", job_id: "sj_cart", schema_version: "studio.analysis.job.v1", status_route: "/api/studio/jobs/sj_cart", job,
    }))));
  let session: EvidenceCartSession | undefined;
  /** Publish the mounted production hook for user-action invocation. */
  function Host() {
    const value = useEvidenceCartSession();
    useEffect(() => { session = value; });
    return null;
  }
  const host = document.createElement("div"); document.body.append(host); root = createRoot(host);
  await act(async () => { root?.render(<StrictMode><Host /></StrictMode>); await Promise.resolve(); });
  if (change) unsubscribe = useStudioStore.subscribe((state, previous) => {
    const field = kind === "simulation" ? "result" : "fiResult";
    if (state[field] !== null && state[field] !== previous[field]) state.setSourceMode("ode");
  });
  await act(async () => {
    if (!session) throw new Error("Cart hook not mounted");
    const run = kind === "simulation" ? session.runSimulationIntoCart : session.runAnalysisIntoCart;
    await Promise.all([run(), run()]);
  });
  expect(session?.cart.items).toHaveLength(1);
  expect(session?.cart.items[0]?.sourceName).toBe("SCLapicqueLIFNeuron");
  expect(session?.cart.items[0]?.payload).toMatchObject({ source_mode: "model" });
  expect(useStudioStore.getState().sourceMode).toBe(change ? "ode" : "model");
  if (!change) {
    await act(async () => {
      if (!session) throw new Error("Cart hook unmounted");
      if (kind === "simulation") await session.runSimulationIntoCart();
      else await session.runAnalysisIntoCart();
    });
    expect(session?.cart.items).toHaveLength(1);
  }
});
