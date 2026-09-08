// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Analysis job completion belongs to a successful current run

import { afterEach, expect, it, vi } from "vitest";
import { studioGuidedFlowInputs } from "../studioGuidedFlowInputs";
import { useStudioStore } from "./studio";

const initial = useStudioStore.getState();

afterEach(() => {
  useStudioStore.setState(initial, true);
  vi.unstubAllGlobals();
});

/** Read the same completion projection as the running application. */
function complete(): boolean {
  return studioGuidedFlowInputs(useStudioStore.getState(), {
    evidenceExportSatisfied: false, trainingSkipped: false,
  }).analysisComplete;
}

/**
 * Release a real HTTP-client request with a terminal job receipt.
 *
 * @param status - Terminal outcome returned by the controlled transport.
 * @param analysis - Analysis response shape exercised by the caller.
 * @returns Function that delivers the response after the store starts its job.
 */
function pendingReceipt(status = "completed", analysis = "fi_curve"): () => void {
  let release = (): void => { throw new Error("request not submitted"); };
  const receipt = {
      analysis, execution_mode: "async_job", job_id: "sj_currency",
      schema_version: "studio.analysis.job.v1", status_route: "/api/studio/jobs/sj_currency",
      job: {
        job_id: "sj_currency", kind: "analysis", status, owner: "studio",
        created_at_utc: "2026-09-08T00:00:00Z", started_at_utc: null,
        finished_at_utc: null, execution_model: "thread", request_id: null,
        artifacts: [], error: status === "completed" ? null : "analysis unavailable",
        result: status === "completed" ? {
          ...(analysis === "heatmap" ? { param_x: "x", param_y: "y", x_values: [1], y_values: [2], rates: [[5]], rate_min: 5, rate_max: 5 }
            : { currents: [0, 1], rates: [0, 5] }),
          analysis_metadata: {
            analysis_type: analysis, evidence_classification: "analysis",
            input_sha256: "a".repeat(64), result_sha256: "b".repeat(64),
            output_keys: analysis === "heatmap" ? ["rates", "x_values", "y_values"] : ["currents", "rates"], schema_version: "studio.analysis-result.v1",
            source: "ode", status: "completed",
          },
        } : null,
      },
  };
  vi.stubGlobal("fetch", vi.fn<typeof globalThis.fetch>((url) => {
    if (url === "/api/studio/jobs/sj_currency") {
      return Promise.resolve(new Response(JSON.stringify(receipt.job), {
        headers: { "Content-Type": "application/json" },
      }));
    }
    return new Promise<Response>((resolve) => {
      release = () => {
        resolve(new Response(JSON.stringify(receipt), {
          headers: { "Content-Type": "application/json" },
        }));
      };
    });
  }));
  return () => { release(); };
}

it.each(["failed", "cancelled", "timed_out"])("does not relabel old results on %s rerun", async (status) => {
  const first = pendingReceipt();
  const run = useStudioStore.getState().runFICurve();
  first();
  await run;
  expect(complete()).toBe(true);
  const oldResult = useStudioStore.getState().fiResult;
  const release = pendingReceipt(status);
  const rerun = useStudioStore.getState().runFICurve();
  const pendingComplete = complete();
  release();
  await rerun;
  expect(useStudioStore.getState().isSimulating).toBe(false);
  expect(useStudioStore.getState().fiResult).toBe(oldResult);
  expect(pendingComplete).toBe(false);
  expect(complete()).toBe(false);
  const recover = pendingReceipt();
  const recovery = useStudioStore.getState().runFICurve();
  recover();
  await recovery;
  expect(complete()).toBe(true);
});

it.each(["selection", "request"] as const)("withdraws completion on invalid %s without submitting", async (failure) => {
  const release = pendingReceipt();
  const run = useStudioStore.getState().runFICurve();
  release();
  await run;
  expect(complete()).toBe(true);
  const oldResult = useStudioStore.getState().fiResult;
  const fetch = vi.fn<typeof globalThis.fetch>();
  vi.stubGlobal("fetch", fetch);
  if (failure === "selection") {
    useStudioStore.getState().setSweepParam("missing_parameter");
    await useStudioStore.getState().runBifurcation();
  } else {
    useStudioStore.setState({ dt: Number.NaN });
    await useStudioStore.getState().runFICurve();
  }
  expect(fetch).not.toHaveBeenCalled();
  expect(useStudioStore.getState().error).not.toBeNull();
  expect(useStudioStore.getState().analysisExperimentKey).toBeNull();
  expect(useStudioStore.getState().fiResult).toBe(oldResult);
  useStudioStore.setState({ dt: initial.dt });
  expect(complete()).toBe(false);
});

it.each(["completed", "failed"])("does not land a late %s analysis on a different source", async (status) => {
  const release = pendingReceipt(status);
  const run = useStudioStore.getState().runFICurve();
  useStudioStore.getState().setSourceMode(initial.sourceMode === "model" ? "ode" : "model");
  release();
  await run;
  expect(useStudioStore.getState().fiResult).toBeNull();
  expect(useStudioStore.getState().analysisExperimentKey).toBeNull();
  expect(useStudioStore.getState().isSimulating).toBe(false);
  expect(complete()).toBe(false);
  expect(useStudioStore.getState().error).toBeNull();
});

it("reports invalid input introduced while the job was running", async () => {
  const release = pendingReceipt();
  const run = useStudioStore.getState().runFICurve();
  useStudioStore.setState({ dt: Number.NaN });
  release();
  await run;
  expect(useStudioStore.getState().fiResult).toBeNull();
  expect(useStudioStore.getState().analysisExperimentKey).toBeNull();
  expect(useStudioStore.getState().isSimulating).toBe(false);
  expect(useStudioStore.getState().error).toContain("NaN");
});

it("attests the heatmap itself and withdraws it on failed rerun", async () => {
  useStudioStore.setState({ sourceMode: "ode", odeParams: { x: 1, y: 2 }, sweepParam: "x", sweepParamY: "y" });
  const release = pendingReceipt("completed", "heatmap");
  const run = useStudioStore.getState().runHeatmap();
  release(); await run;
  expect(useStudioStore.getState().heatmapResult).not.toBeNull();
  expect(useStudioStore.getState().heatmapExperimentKey).toBe(useStudioStore.getState().analysisExperimentKey);
  expect(useStudioStore.getState().heatmapExperimentKey).not.toBeNull();
  const old = useStudioStore.getState().heatmapResult;
  const fail = pendingReceipt("failed", "heatmap");
  const rerun = useStudioStore.getState().runHeatmap();
  expect(useStudioStore.getState().heatmapExperimentKey).toBeNull();
  fail(); await rerun;
  expect(useStudioStore.getState().heatmapExperimentKey).toBeNull();
  expect(useStudioStore.getState().heatmapResult).toBe(old);
});
