// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained Training Monitor recovery tests

import { afterEach, describe, expect, it, vi } from "vitest";

import { setStudioAuthToken } from "./api/client";
import { useStudioStore } from "./stores/studio";
import {
  decodeTrainingJobSummaries,
  decodeTrainingRecoveryStatus,
  decodeTrainingStopResult,
} from "./studioTrainingRecovery";

const retainedConfig = {
  schema_version: "studio.training-config.v1",
  dataset: "mnist",
  epochs: 4,
  batch_size: 32,
  lr: 0.001,
  hidden: [64],
  timesteps: 12,
  surrogate: "atan_surrogate",
  learn_beta: false,
  learn_threshold: false,
  max_grad_norm: 1,
  seed: 7,
};
const originalState = useStudioStore.getState();

afterEach(() => {
  setStudioAuthToken(null);
  useStudioStore.setState(originalState, true);
  vi.unstubAllGlobals();
});

describe("retained Training Monitor recovery", () => {
  it("rejects malformed provenance and a status from another job", () => {
    expect(decodeTrainingJobSummaries([
      { job_id: "sj_old", status: "interrupted", config: null },
    ])[0]?.config).toBeNull();
    expect(decodeTrainingJobSummaries([
      { job_id: "sj_direct", status: "completed", config: { ...retainedConfig, hidden: [], max_grad_norm: 0 } },
    ])[0]?.config?.hidden).toEqual([]);
    expect(() => decodeTrainingJobSummaries([
      { job_id: "sj_bad", status: "completed", config: { ...retainedConfig, epochs: -1 } },
    ])).toThrow("invalid configuration");
    expect(() => decodeTrainingRecoveryStatus(
      { job_id: "sj_other", status: "completed" }, "sj_selected",
    )).toThrow("invalid");
    expect(() => decodeTrainingStopResult(
      { job_id: "sj_selected", status: "cancelled" }, "sj_selected",
    )).toThrow("invalid");
    expect(decodeTrainingStopResult(
      { job_id: "sj_selected", status: "unknown" }, "sj_selected",
    )).toBe("unknown");
  });

  it("loads the retained list, selects the newest job and replays authenticated SSE", async () => {
    useStudioStore.setState({
      trainingJobId: null,
      trainingStatus: "idle",
      trainingEpochs: [],
      trainingJobs: [],
      trainingObservedConfig: null,
    });
    const editableConfig = useStudioStore.getState().trainingConfig;
    setStudioAuthToken("viewer-token");
    const fetcher = vi.fn(async (input: string | URL | Request, _init?: RequestInit) => {
      if (typeof input !== "string") throw new Error("Expected a relative Studio route");
      const path = input;
      if (path === "/api/training/jobs") {
        return new Response(JSON.stringify([
          { job_id: "sj_old", status: "interrupted", config: null },
          { job_id: "sj_recent", status: "completed", config: retainedConfig },
        ]), { status: 200 });
      }
      if (path === "/api/training/status/sj_recent") {
        return new Response(JSON.stringify({ job_id: "sj_recent", status: "completed" }), { status: 200 });
      }
      if (path === "/api/training/status/sj_old") {
        return new Response(JSON.stringify({ job_id: "sj_old", status: "interrupted" }), { status: 200 });
      }
      if (path === "/api/training/stream/sj_recent") {
        return new Response('data: {"event":"epoch","data":{"epoch":1,"train_loss":0.2,"train_accuracy":0.8,"val_loss":0.3,"val_accuracy":0.7}}\n\ndata: {"event":"completed"}\n\n', {
          headers: { "content-type": "text/event-stream" },
        });
      }
      if (path === "/api/training/stream/sj_old") {
        return new Response('data: {"event":"interrupted"}\n\n', {
          headers: { "content-type": "text/event-stream" },
        });
      }
      throw new Error(`Unexpected route ${path}`);
    });
    vi.stubGlobal("fetch", fetcher);

    await useStudioStore.getState().loadTrainingJobs();
    await vi.waitFor(() => {
      expect(useStudioStore.getState().trainingEpochs.map((epoch) => epoch.epoch)).toEqual([1]);
    });

    const state = useStudioStore.getState();
    expect(state.trainingJobs.map((job) => job.job_id)).toEqual(["sj_old", "sj_recent"]);
    expect(state.trainingJobId).toBe("sj_recent");
    expect(state.trainingStatus).toBe("completed");
    expect(state.trainingObservedConfig?.dataset).toBe("mnist");
    expect(state.trainingConfig).toEqual(editableConfig);
    expect(state.trainingExperimentKey).toBeNull();
    await useStudioStore.getState().selectTrainingJob("sj_old");
    await vi.waitFor(() => { expect(useStudioStore.getState().trainingStatus).toBe("interrupted"); });
    expect(useStudioStore.getState().trainingObservedConfig).toBeNull();
    expect(useStudioStore.getState().trainingConfig).toEqual(editableConfig);
    for (const [path, init] of fetcher.mock.calls) {
      expect(init?.headers).toMatchObject({ Authorization: "Bearer viewer-token" });
      expect(path).toMatch(/^\/api\/training\//);
    }
  });

  it("reconnects the selected job after a transport failure when runs are refreshed", async () => {
    useStudioStore.setState({
      trainingJobId: null, trainingStatus: "idle", trainingEpochs: [],
      trainingJobs: [], trainingObservedConfig: null,
    });
    let streamRequests = 0;
    const fetcher = vi.fn(async (input: string | URL | Request) => {
      if (typeof input !== "string") throw new Error("Expected a relative Studio route");
      if (input === "/api/training/jobs") {
        return new Response(JSON.stringify([
          { job_id: "sj_running", status: "running", config: retainedConfig },
        ]), { status: 200 });
      }
      if (input === "/api/training/status/sj_running") {
        return new Response(JSON.stringify({ job_id: "sj_running", status: "running" }), { status: 200 });
      }
      if (input === "/api/training/stream/sj_running") {
        streamRequests += 1;
        return streamRequests === 1
          ? new Response(null, { status: 503 })
          : new Response('data: {"event":"completed"}\n\n', {
            headers: { "content-type": "text/event-stream" },
          });
      }
      throw new Error(`Unexpected route ${input}`);
    });
    vi.stubGlobal("fetch", fetcher);

    await useStudioStore.getState().loadTrainingJobs();
    await vi.waitFor(() => { expect(useStudioStore.getState().trainingStatus).toBe("disconnected"); });
    await useStudioStore.getState().loadTrainingJobs();
    await vi.waitFor(() => { expect(useStudioStore.getState().trainingStatus).toBe("completed"); });

    expect(useStudioStore.getState().trainingJobId).toBe("sj_running");
    expect(streamRequests).toBe(2);
  });

  it("keeps a terminal Stop response instead of falsely reporting stopping", async () => {
    useStudioStore.setState({ trainingJobId: "sj_finished", trainingStatus: "unknown" });
    const fetcher = vi.fn(async (_input: string | URL | Request, _init?: RequestInit) => new Response(JSON.stringify({
      job_id: "sj_finished", status: "completed",
    }), { status: 200 }));
    vi.stubGlobal("fetch", fetcher);

    await useStudioStore.getState().stopTraining();

    expect(useStudioStore.getState().trainingStatus).toBe("completed");
    expect(fetcher.mock.calls[0]?.[0]).toBe("/api/training/stop");
  });

  it("does not let a delayed Stop reply overwrite a newer terminal event", async () => {
    useStudioStore.setState({ trainingJobId: "sj_finished", trainingStatus: "running" });
    let answer: (response: Response) => void = () => { throw new Error("Stop request missing"); };
    const fetcher = vi.fn(() => new Promise<Response>((resolve) => { answer = resolve; }));
    vi.stubGlobal("fetch", fetcher);

    const pending = useStudioStore.getState().stopTraining();
    useStudioStore.setState({ trainingStatus: "completed" });
    answer(new Response(JSON.stringify({ job_id: "sj_finished", status: "stopping" })));
    await pending;

    expect(useStudioStore.getState().trainingStatus).toBe("completed");
  });
});
