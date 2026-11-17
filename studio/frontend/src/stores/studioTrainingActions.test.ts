// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio training actions against a mocked server, event stream and clock

import { createStore } from "zustand/vanilla";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  attachTrainingWeights,
  attachTrainingWeightsLive,
  exportTrainingCheckpoint,
  fetchStudioJobArtifact,
  fetchSurrogates,
  fetchTargetProfiles,
  fetchTrainingJobs,
  fetchTrainingStatus,
  importTrainingCheckpoint,
  logoutStudioBrowserUser,
  restoreTrainingWeights,
  startTraining,
  stopTraining,
} from "../api/client";
import { connectStudioTrainingEventSource } from "../studioTrainingStream";
import { parseTrainingCheckpointPayload } from "../trainingCheckpoint";
import { trainingCheckpointExportPlan, trainingWeightRestoreVerificationExportPlan } from "../trainingExports";
import { readTrainingTargetProfiles } from "../trainingRequest";
import { verifyTrainingWeightArtifactBlob } from "../trainingRestore";
import {
  trainingCheckpointImportedState,
  trainingSurrogatesLoadedState,
  trainingWeightLiveAttachLoadedState,
  trainingWeightMaterializationLoadedState,
  trainingWeightRestoreVerificationLoadedState,
} from "../trainingStoreState";
import { studioInitialData } from "./studioInitialState";
import { createStudioStoreActions } from "./studioStoreActions";
import { createStudioEquationIrActions } from "./studioEquationIrActions";
import { createStudioTrainingActions, type StudioTrainingActionBinding } from "./studioTrainingActions";
import type { StudioState } from "./studioTypes";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    attachTrainingWeights: vi.fn(),
    attachTrainingWeightsLive: vi.fn(),
    exportTrainingCheckpoint: vi.fn(),
    fetchStudioJobArtifact: vi.fn(),
    fetchSurrogates: vi.fn(),
    fetchTargetProfiles: vi.fn(),
    fetchTrainingJobs: vi.fn(),
    fetchTrainingStatus: vi.fn(),
    importTrainingCheckpoint: vi.fn(),
    logoutStudioBrowserUser: vi.fn(),
    restoreTrainingWeights: vi.fn(),
    startTraining: vi.fn(),
    stopTraining: vi.fn(),
  };
});
vi.mock("../studioTrainingStream", async () => {
  const actual = await vi.importActual<typeof import("../studioTrainingStream")>("../studioTrainingStream");
  return { ...actual, connectStudioTrainingEventSource: vi.fn() };
});
vi.mock("../trainingCheckpoint", async () => {
  const actual = await vi.importActual<typeof import("../trainingCheckpoint")>("../trainingCheckpoint");
  return { ...actual, parseTrainingCheckpointPayload: vi.fn() };
});
vi.mock("../trainingExports", async () => {
  const actual = await vi.importActual<typeof import("../trainingExports")>("../trainingExports");
  return { ...actual, trainingCheckpointExportPlan: vi.fn(), trainingWeightRestoreVerificationExportPlan: vi.fn() };
});
vi.mock("../trainingRequest", async () => {
  const actual = await vi.importActual<typeof import("../trainingRequest")>("../trainingRequest");
  return { ...actual, readTrainingTargetProfiles: vi.fn() };
});
vi.mock("../trainingRestore", async () => {
  const actual = await vi.importActual<typeof import("../trainingRestore")>("../trainingRestore");
  return { ...actual, verifyTrainingWeightArtifactBlob: vi.fn() };
});

afterEach(() => {
  vi.useRealTimers();
  vi.resetAllMocks();
});

type StreamHandlers = Parameters<typeof connectStudioTrainingEventSource>[1];

/** The event streams the actions opened, newest last, each with its handlers and its close. */
const streams: { jobId: string; handlers: StreamHandlers; close: ReturnType<typeof vi.fn> }[] = [];

/** Record every stream the actions open instead of connecting. */
function recordStreams(): void {
  streams.length = 0;
  vi.mocked(connectStudioTrainingEventSource).mockImplementation((jobId: string, handlers: StreamHandlers) => {
    const close = vi.fn();
    streams.push({ jobId, handlers, close });
    return { close } as unknown as ReturnType<typeof connectStudioTrainingEventSource>;
  });
}

/** The newest stream's handlers. */
function latestStream() {
  const stream = streams.at(-1);
  if (stream === undefined) throw new Error("no stream was opened");
  return stream;
}

/** A store with the real actions over the initial data; the training binding is kept for its release hook. */
function studioStore() {
  const held: { binding?: StudioTrainingActionBinding } = {};
  const store = createStore<StudioState>((set, get) => {
    const binding = createStudioTrainingActions(set, get);
    held.binding = binding;
    return {
      ...studioInitialData, ...createStudioStoreActions(set, get), ...createStudioEquationIrActions(set, get),
      ...binding.actions,
    };
  });
  if (held.binding === undefined) throw new Error("the store was not built");
  return { store, release: held.binding.releaseTrainingSession };
}

/** A promise the test settles. */
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((ok, fail) => { resolve = ok; reject = fail; });
  return { promise, resolve, reject };
}

/** A server answer of the type the mocked function resolves to. */
function answer<F extends (...args: never[]) => Promise<unknown>>(_: F, value: object): Awaited<ReturnType<F>> {
  return value as Awaited<ReturnType<F>>;
}

const job = (id: string, status: string) => ({ job_id: id, status, config: null });
const EARLIER_VERDICT = { verdict: "earlier" } as unknown as StudioState["trainingPreregistrationVerdict"];
const EARLIER_CONVERSION = { result: "earlier" } as unknown as StudioState["trainingConversionResult"];
const status = (id: string, value: string) =>
  answer(fetchTrainingStatus, { job_id: id, status: value, preregistration_verdict: null, final_metrics: null });

/** A store following job `id`, selected through the real action, with its stream recorded. */
async function following(id = "sj_1", value = "running") {
  recordStreams();
  const made = studioStore();
  made.store.setState({ trainingJobs: [job(id, value)] as StudioState["trainingJobs"] });
  vi.mocked(fetchTrainingStatus).mockResolvedValueOnce(status(id, value));
  await made.store.getState().selectTrainingJob(id);
  return made;
}

describe("surrogates and target profiles", () => {
  it("loads the surrogates, and treats a failure as not critical", async () => {
    const { store } = studioStore();
    const surrogates = answer(fetchSurrogates, [{ name: "fast_sigmoid" }]);
    vi.mocked(fetchSurrogates).mockResolvedValueOnce(surrogates).mockRejectedValueOnce(new Error("offline"));
    await store.getState().loadSurrogates();
    expect(store.getState()).toMatchObject(trainingSurrogatesLoadedState(surrogates));
    await store.getState().loadSurrogates();
    expect(store.getState().error).toBeNull();
  });

  it("loads the target profiles the request reader accepts, and offers none on a failure", async () => {
    const { store } = studioStore();
    vi.mocked(fetchTargetProfiles).mockResolvedValueOnce(answer(fetchTargetProfiles, {}));
    vi.mocked(readTrainingTargetProfiles).mockReturnValueOnce(["loihi2"] as unknown as ReturnType<typeof readTrainingTargetProfiles>);
    await store.getState().loadTargetProfiles();
    expect(store.getState().trainingTargetProfiles).toEqual(["loihi2"]);
    vi.mocked(fetchTargetProfiles).mockRejectedValueOnce(new Error("offline"));
    await store.getState().loadTargetProfiles();
    expect(store.getState().trainingTargetProfiles).toEqual(["loihi2"]);
    expect(store.getState().error).toBeNull();
  });
});

describe("the job list", () => {
  it("selects the newest job when nothing is selected and nothing runs", async () => {
    const { store } = studioStore();
    const selectTrainingJob = vi.fn(() => Promise.resolve());
    store.setState({ selectTrainingJob });
    vi.mocked(fetchTrainingJobs).mockResolvedValue(answer(fetchTrainingJobs, [job("sj_1", "completed"), job("sj_2", "running")]));
    await store.getState().loadTrainingJobs();
    expect(store.getState().trainingJobs.map((row) => row.job_id)).toEqual(["sj_1", "sj_2"]);
    expect(selectTrainingJob).toHaveBeenCalledWith("sj_2");
  });

  it("follows the disconnected job again once the list holds it", async () => {
    const { store } = studioStore();
    const selectTrainingJob = vi.fn(() => Promise.resolve());
    store.setState({ selectTrainingJob, trainingJobId: "sj_1", trainingStatus: "disconnected" });
    vi.mocked(fetchTrainingJobs).mockResolvedValue(answer(fetchTrainingJobs, [job("sj_1", "running")]));
    await store.getState().loadTrainingJobs();
    expect(selectTrainingJob).toHaveBeenCalledWith("sj_1");
  });

  it("selects nothing when a job is already followed", async () => {
    const { store } = studioStore();
    const selectTrainingJob = vi.fn(() => Promise.resolve());
    store.setState({ selectTrainingJob, trainingJobId: "sj_1", trainingStatus: "running" });
    vi.mocked(fetchTrainingJobs).mockResolvedValue(answer(fetchTrainingJobs, [job("sj_1", "running")]));
    await store.getState().loadTrainingJobs();
    expect(selectTrainingJob).not.toHaveBeenCalled();
  });

  it("keeps only the newest of two overlapping list replies, success or failure", async () => {
    const { store } = studioStore();
    const first = deferred<Awaited<ReturnType<typeof fetchTrainingJobs>>>();
    vi.mocked(fetchTrainingJobs)
      .mockReturnValueOnce(first.promise)
      .mockResolvedValueOnce(answer(fetchTrainingJobs, [job("sj_new", "completed")]));
    const late = store.getState().loadTrainingJobs();
    await store.getState().loadTrainingJobs();
    first.resolve(answer(fetchTrainingJobs, [job("sj_old", "completed")]));
    await late;
    expect(store.getState().trainingJobs.map((row) => row.job_id)).toEqual(["sj_new"]);
    const stale = deferred<Awaited<ReturnType<typeof fetchTrainingJobs>>>();
    vi.mocked(fetchTrainingJobs).mockReset();
    vi.mocked(fetchTrainingJobs).mockReturnValueOnce(stale.promise).mockResolvedValueOnce(answer(fetchTrainingJobs, []));
    const staleLoad = store.getState().loadTrainingJobs();
    await store.getState().loadTrainingJobs();
    stale.reject(new Error("late failure"));
    await staleLoad;
    expect(store.getState().trainingJobsError).toBeNull();
  });

  it("reports a list the server refuses", async () => {
    const { store } = studioStore();
    vi.mocked(fetchTrainingJobs).mockRejectedValue(new Error("refused"));
    await store.getState().loadTrainingJobs();
    expect(store.getState().trainingJobsError).toBe("Error: refused");
    expect(store.getState().trainingJobsLoading).toBe(false);
  });
});

describe("selecting a job", () => {
  it("refuses a job the list does not retain", async () => {
    const { store } = studioStore();
    await store.getState().selectTrainingJob("gone");
    expect(store.getState().trainingJobsError).toBe("Training job is not in the retained list");
    expect(fetchTrainingStatus).not.toHaveBeenCalled();
  });

  it("recovers the job's status and follows its event stream", async () => {
    const { store } = await following("sj_1", "running");
    expect(store.getState().trainingJobId).toBe("sj_1");
    expect(store.getState().trainingStatus).toBe("running");
    expect(latestStream().jobId).toBe("sj_1");
  });

  it("keeps only the newest of two overlapping selections, success or failure", async () => {
    const { store } = studioStore();
    recordStreams();
    store.setState({ trainingJobs: [job("sj_1", "running"), job("sj_2", "completed")] as StudioState["trainingJobs"] });
    const first = deferred<Awaited<ReturnType<typeof fetchTrainingStatus>>>();
    vi.mocked(fetchTrainingStatus).mockReturnValueOnce(first.promise).mockResolvedValueOnce(status("sj_2", "completed"));
    const late = store.getState().selectTrainingJob("sj_1");
    await store.getState().selectTrainingJob("sj_2");
    first.resolve(status("sj_1", "running"));
    await late;
    expect(store.getState().trainingJobId).toBe("sj_2");
    const stale = deferred<Awaited<ReturnType<typeof fetchTrainingStatus>>>();
    vi.mocked(fetchTrainingStatus).mockReturnValueOnce(stale.promise).mockResolvedValueOnce(status("sj_2", "completed"));
    const staleSelect = store.getState().selectTrainingJob("sj_1");
    await store.getState().selectTrainingJob("sj_2");
    stale.reject(new Error("late"));
    await staleSelect;
    expect(store.getState().trainingJobsError).toBeNull();
  });

  it("reports a status the server refuses", async () => {
    const { store } = studioStore();
    store.setState({ trainingJobs: [job("sj_1", "running")] as StudioState["trainingJobs"] });
    vi.mocked(fetchTrainingStatus).mockRejectedValue(new Error("refused"));
    await store.getState().selectTrainingJob("sj_1");
    expect(store.getState().trainingJobsError).toBe("Error: refused");
  });
});

describe("the event stream", () => {
  it("appends each epoch of the followed job, and none after another job is followed", async () => {
    const { store } = await following();
    const metrics = { epoch: 1 } as unknown as Parameters<StreamHandlers["onEpoch"]>[0];
    latestStream().handlers.onEpoch(metrics);
    expect(store.getState().trainingEpochs).toEqual([metrics]);
    store.setState({ trainingJobId: "sj_other" });
    latestStream().handlers.onEpoch(metrics);
    expect(store.getState().trainingEpochs).toEqual([metrics]);
  });

  it("marks a running job disconnected, but not a finished one", async () => {
    const { store } = await following();
    latestStream().handlers.onDisconnected();
    expect(store.getState().trainingStatus).toBe("disconnected");
    store.setState({ trainingStatus: "completed" });
    latestStream().handlers.onDisconnected();
    expect(store.getState().trainingStatus).toBe("completed");
  });

  it("reports a stream error for a running job and lists the jobs again", async () => {
    const { store } = await following();
    vi.mocked(fetchTrainingJobs).mockResolvedValue(answer(fetchTrainingJobs, [job("sj_1", "running")]));
    latestStream().handlers.onError("worker lost");
    expect(store.getState().error).toBe("worker lost");
    expect(store.getState().trainingStatus).toBe("failed");
    await vi.waitFor(() => { expect(fetchTrainingJobs).toHaveBeenCalled(); });
  });

  it("only lists the jobs again on a stream error after the job finished, and ignores an outdated stream", async () => {
    const { store } = await following();
    vi.mocked(fetchTrainingJobs).mockResolvedValue(answer(fetchTrainingJobs, [job("sj_1", "completed")]));
    store.setState({ trainingStatus: "completed" });
    latestStream().handlers.onError("late");
    expect(store.getState().error).toBeNull();
    await vi.waitFor(() => { expect(fetchTrainingJobs).toHaveBeenCalledOnce(); });
    store.setState({ trainingJobId: "sj_other" });
    latestStream().handlers.onError("outdated");
    latestStream().handlers.onTerminal("stopped");
    expect(store.getState().error).toBeNull();
    expect(fetchTrainingJobs).toHaveBeenCalledOnce();
  });

  it("reads a completed job's sealed verdict and lists until the list agrees", async () => {
    vi.useFakeTimers();
    const { store } = await following();
    vi.mocked(fetchTrainingStatus)
      .mockResolvedValueOnce(status("sj_1", "running"))
      .mockResolvedValueOnce(status("sj_1", "completed"));
    vi.mocked(fetchTrainingJobs)
      .mockResolvedValueOnce(answer(fetchTrainingJobs, [job("sj_1", "running")]))
      .mockResolvedValueOnce(answer(fetchTrainingJobs, [job("sj_1", "completed")]));
    store.setState({ trainingPreregistrationVerdict: EARLIER_VERDICT, trainingConversionResult: EARLIER_CONVERSION });
    latestStream().handlers.onTerminal("completed");
    expect(store.getState().trainingStatus).toBe("completed");
    await vi.advanceTimersByTimeAsync(1000);
    expect(fetchTrainingStatus).toHaveBeenCalledTimes(3);
    expect(fetchTrainingJobs).toHaveBeenCalledTimes(2);
    // The sealed record carries neither, so both are read as none and replace the earlier ones.
    expect(store.getState().trainingPreregistrationVerdict).toBeNull();
    expect(store.getState().trainingConversionResult).toBeNull();
  });

  it("does not read a verdict for a job that ended without completing, nor stop on a job the list dropped", async () => {
    vi.useFakeTimers();
    const { store } = await following();
    vi.mocked(fetchTrainingStatus).mockResolvedValueOnce(status("sj_1", "stopped"));
    vi.mocked(fetchTrainingJobs).mockResolvedValueOnce(answer(fetchTrainingJobs, []));
    store.setState({ trainingStatus: "stopped", trainingPreregistrationVerdict: EARLIER_VERDICT });
    latestStream().handlers.onTerminal("stopped");
    await vi.advanceTimersByTimeAsync(10);
    expect(store.getState().trainingStatus).toBe("stopped");
    expect(store.getState().trainingPreregistrationVerdict).toBe(EARLIER_VERDICT);
    expect(fetchTrainingJobs).toHaveBeenCalledOnce();
  });

  it("lists what the server says once the job never seals within the reads", async () => {
    vi.useFakeTimers();
    const { store } = await following();
    vi.mocked(fetchTrainingStatus).mockResolvedValue(status("sj_1", "running"));
    vi.mocked(fetchTrainingJobs).mockResolvedValue(answer(fetchTrainingJobs, [job("sj_1", "running")]));
    latestStream().handlers.onTerminal("completed");
    await vi.advanceTimersByTimeAsync(250 * 41);
    expect(fetchTrainingStatus).toHaveBeenCalledTimes(41);
    expect(fetchTrainingJobs).toHaveBeenCalled();
    expect(store.getState().trainingStatus).toBe("completed");
  });

  it("reports a verdict read the server refuses", async () => {
    const { store } = await following();
    vi.mocked(fetchTrainingStatus).mockRejectedValueOnce(new Error("refused"));
    latestStream().handlers.onTerminal("completed");
    await vi.waitFor(() => { expect(store.getState().trainingJobsError).toBe("Error: refused"); });
  });

  it("drops a verdict read whose job is no longer followed", async () => {
    const { store } = await following();
    const late = deferred<Awaited<ReturnType<typeof fetchTrainingStatus>>>();
    vi.mocked(fetchTrainingStatus).mockReturnValueOnce(late.promise);
    latestStream().handlers.onTerminal("completed");
    store.setState({ trainingJobId: "sj_other" });
    late.reject(new Error("late"));
    await vi.waitFor(() => { expect(fetchTrainingStatus).toHaveBeenCalledTimes(2); });
    expect(store.getState().trainingJobsError).toBeNull();
  });
});

describe("starting and stopping", () => {
  it("starts no second run while one is running", async () => {
    const { store } = studioStore();
    store.setState({ trainingStatus: "running" });
    await store.getState().startTraining();
    expect(startTraining).not.toHaveBeenCalled();
  });

  it("starts a run, follows it and lists the jobs", async () => {
    const { store } = studioStore();
    recordStreams();
    vi.mocked(startTraining).mockResolvedValue(answer(startTraining, { job_id: "sj_new" }));
    vi.mocked(fetchTrainingJobs).mockResolvedValue(answer(fetchTrainingJobs, [job("sj_new", "running")]));
    await store.getState().startTraining();
    expect(store.getState().trainingJobId).toBe("sj_new");
    expect(store.getState().trainingStatus).toBe("running");
    expect(latestStream().jobId).toBe("sj_new");
    await vi.waitFor(() => { expect(fetchTrainingJobs).toHaveBeenCalled(); });
  });

  it("drops the reply of a selection made before the start", async () => {
    const { store } = studioStore();
    recordStreams();
    store.setState({ trainingJobs: [job("sj_1", "completed")] as StudioState["trainingJobs"] });
    const selected = deferred<Awaited<ReturnType<typeof fetchTrainingStatus>>>();
    vi.mocked(fetchTrainingStatus).mockReturnValueOnce(selected.promise);
    vi.mocked(startTraining).mockResolvedValue(answer(startTraining, { job_id: "sj_new" }));
    vi.mocked(fetchTrainingJobs).mockResolvedValue(answer(fetchTrainingJobs, []));
    const selecting = store.getState().selectTrainingJob("sj_1");
    await store.getState().startTraining();
    selected.resolve(status("sj_1", "completed"));
    await selecting;
    expect(store.getState().trainingJobId).toBe("sj_new");
    expect(store.getState().trainingStatus).toBe("running");
  });

  it("only lists the jobs when the start was outdated while it ran", async () => {
    const { store, release } = studioStore();
    recordStreams();
    const pending = deferred<Awaited<ReturnType<typeof startTraining>>>();
    vi.mocked(startTraining).mockReturnValue(pending.promise);
    vi.mocked(fetchTrainingJobs).mockResolvedValue(answer(fetchTrainingJobs, []));
    const start = store.getState().startTraining();
    release();
    pending.resolve(answer(startTraining, { job_id: "sj_new" }));
    await start;
    expect(store.getState().trainingJobId).toBeNull();
    expect(streams).toEqual([]);
    expect(fetchTrainingJobs).toHaveBeenCalled();
  });

  it("marks a refused start failed, unless it was outdated", async () => {
    const { store, release } = studioStore();
    vi.mocked(startTraining).mockRejectedValueOnce(new Error("no dataset"));
    await store.getState().startTraining();
    expect(store.getState().error).toBe("no dataset");
    expect(store.getState().trainingStatus).toBe("failed");
    store.setState({ error: null, trainingStatus: "idle" });
    const pending = deferred<Awaited<ReturnType<typeof startTraining>>>();
    vi.mocked(startTraining).mockReturnValueOnce(pending.promise);
    const start = store.getState().startTraining();
    release();
    pending.reject(new Error("late"));
    await start;
    expect(store.getState().error).toBeNull();
  });

  it("stops nothing when no job is followed", async () => {
    const { store } = studioStore();
    await store.getState().stopTraining();
    expect(stopTraining).not.toHaveBeenCalled();
  });

  it("writes the stop outcome for a running job, and leaves a finished one as it is", async () => {
    const { store } = studioStore();
    store.setState({ trainingJobId: "sj_1", trainingStatus: "running" });
    vi.mocked(stopTraining).mockResolvedValue(answer(stopTraining, { job_id: "sj_1", status: "stopping" }));
    await store.getState().stopTraining();
    expect(store.getState().trainingStatus).toBe("stopping");
    store.setState({ trainingStatus: "completed" });
    await store.getState().stopTraining();
    expect(store.getState().trainingStatus).toBe("completed");
  });

  it("drops a stop outcome or refusal once another job is followed", async () => {
    const { store } = studioStore();
    store.setState({ trainingJobId: "sj_1", trainingStatus: "running" });
    const outcome = deferred<Awaited<ReturnType<typeof stopTraining>>>();
    vi.mocked(stopTraining).mockReturnValueOnce(outcome.promise);
    const stop = store.getState().stopTraining();
    store.setState({ trainingJobId: "sj_2" });
    outcome.resolve(answer(stopTraining, { job_id: "sj_1", status: "stopping" }));
    await stop;
    expect(store.getState().trainingStatus).toBe("running");
    store.setState({ trainingJobId: "sj_1" });
    const refusal = deferred<Awaited<ReturnType<typeof stopTraining>>>();
    vi.mocked(stopTraining).mockReturnValueOnce(refusal.promise);
    const refused = store.getState().stopTraining();
    store.setState({ trainingJobId: "sj_2" });
    refusal.reject(new Error("late"));
    await refused;
    expect(store.getState().error).toBeNull();
  });

  it("reports a stop the server refuses", async () => {
    const { store } = studioStore();
    store.setState({ trainingJobId: "sj_1", trainingStatus: "running" });
    vi.mocked(stopTraining).mockRejectedValue(new Error(""));
    await store.getState().stopTraining();
    expect(store.getState().error).toBe("Training stop failed");
  });
});

describe("checkpoints", () => {
  it("exports nothing without a job, writes the checkpoint of one, and reports a refusal", async () => {
    const { store } = studioStore();
    await store.getState().exportTrainingCheckpoint();
    expect(exportTrainingCheckpoint).not.toHaveBeenCalled();
    store.setState({ trainingJobId: "sj_1" });
    const writeExport = vi.fn();
    vi.mocked(exportTrainingCheckpoint).mockResolvedValueOnce(answer(exportTrainingCheckpoint, {}));
    vi.mocked(trainingCheckpointExportPlan).mockReturnValueOnce({ writeExport } as unknown as ReturnType<typeof trainingCheckpointExportPlan>);
    await store.getState().exportTrainingCheckpoint();
    expect(writeExport).toHaveBeenCalledOnce();
    vi.mocked(exportTrainingCheckpoint).mockRejectedValueOnce(new Error(""));
    await store.getState().exportTrainingCheckpoint();
    expect(store.getState().error).toBe("Training checkpoint export failed");
  });

  it("imports a checkpoint into the training configuration, and reports one it cannot read", async () => {
    const { store } = studioStore();
    const imported = answer(importTrainingCheckpoint, { config: { epochs: 3 } });
    vi.mocked(parseTrainingCheckpointPayload).mockReturnValueOnce({} as ReturnType<typeof parseTrainingCheckpointPayload>);
    vi.mocked(importTrainingCheckpoint).mockResolvedValueOnce(imported);
    const before = store.getState().trainingConfig;
    await store.getState().importTrainingCheckpointText("{}");
    expect(store.getState()).toMatchObject(trainingCheckpointImportedState(before, imported));
    vi.mocked(parseTrainingCheckpointPayload).mockImplementationOnce(() => { throw new Error(""); });
    await store.getState().importTrainingCheckpointText("not json");
    expect(store.getState().error).toBe("Training checkpoint import failed");
  });
});

describe("restored weights", () => {
  const PLAN = { source_job_id: "sj_1", weights_artifact: { relative_path: "weights.npz" } };

  it("verifies nothing without a restore plan", async () => {
    const { store } = studioStore();
    await store.getState().verifyTrainingWeightRestoreArtifact();
    expect(store.getState().error).toBe("No training weight restore plan is available.");
  });

  it("verifies the planned artifact, and reports a failed verification", async () => {
    const { store } = studioStore();
    store.setState({ trainingWeightRestorePlan: PLAN as unknown as StudioState["trainingWeightRestorePlan"] });
    const verification = { verified: true } as unknown as Awaited<ReturnType<typeof verifyTrainingWeightArtifactBlob>>;
    vi.mocked(fetchStudioJobArtifact).mockResolvedValue(answer(fetchStudioJobArtifact, {}));
    vi.mocked(verifyTrainingWeightArtifactBlob).mockResolvedValueOnce(verification);
    await store.getState().verifyTrainingWeightRestoreArtifact();
    expect(fetchStudioJobArtifact).toHaveBeenCalledWith("sj_1", "weights.npz");
    expect(store.getState()).toMatchObject(trainingWeightRestoreVerificationLoadedState(verification));
    vi.mocked(verifyTrainingWeightArtifactBlob).mockRejectedValueOnce(new Error(""));
    await store.getState().verifyTrainingWeightRestoreArtifact();
    expect(store.getState().error).toBe("Training weight artifact verification failed");
  });

  it("exports a verification only when there is one, and reports a write that fails", () => {
    const { store } = studioStore();
    vi.mocked(trainingWeightRestoreVerificationExportPlan).mockReturnValueOnce({ available: false, message: "Verify first." } as unknown as ReturnType<typeof trainingWeightRestoreVerificationExportPlan>);
    store.getState().exportTrainingWeightRestoreVerification();
    expect(store.getState().error).toBe("Verify first.");
    const writeExport = vi.fn();
    vi.mocked(trainingWeightRestoreVerificationExportPlan).mockReturnValueOnce({ available: true, writeExport } as unknown as ReturnType<typeof trainingWeightRestoreVerificationExportPlan>);
    store.getState().exportTrainingWeightRestoreVerification();
    expect(writeExport).toHaveBeenCalledOnce();
    expect(store.getState().error).toBeNull();
    vi.mocked(trainingWeightRestoreVerificationExportPlan).mockReturnValueOnce({ available: true, writeExport: () => { throw new Error(""); } } as unknown as ReturnType<typeof trainingWeightRestoreVerificationExportPlan>);
    store.getState().exportTrainingWeightRestoreVerification();
    expect(store.getState().error).toBe("Training weight restore verification export failed");
  });

  it("materializes a job's weights, and asks for a job first", async () => {
    const { store } = studioStore();
    await store.getState().materializeTrainingWeights();
    expect(store.getState().error).toBe("No completed training job is available.");
    store.setState({ trainingJobId: "sj_1", error: null });
    const materialization = answer(restoreTrainingWeights, { source_job_id: "sj_1" });
    vi.mocked(restoreTrainingWeights).mockResolvedValueOnce(materialization).mockRejectedValueOnce(new Error(""));
    await store.getState().materializeTrainingWeights();
    expect(store.getState()).toMatchObject(trainingWeightMaterializationLoadedState(materialization));
    await store.getState().materializeTrainingWeights();
    expect(store.getState().error).toBe("Training weight materialization failed");
  });

  it("attaches weights live to a running job from a verified source, and asks for both first", async () => {
    const { store } = studioStore();
    await store.getState().liveAttachTrainingWeights();
    expect(store.getState().error).toBe("A running target job and a verified source are required.");
    store.setState({
      trainingJobId: "sj_2",
      trainingWeightMaterialization: { source_job_id: "sj_1" } as unknown as StudioState["trainingWeightMaterialization"],
    });
    const attached = answer(attachTrainingWeightsLive, { attached: true });
    vi.mocked(attachTrainingWeightsLive).mockResolvedValueOnce(attached).mockRejectedValueOnce(new Error(""));
    await store.getState().liveAttachTrainingWeights();
    expect(attachTrainingWeightsLive).toHaveBeenCalledWith("sj_2", "sj_1");
    expect(store.getState()).toMatchObject(trainingWeightLiveAttachLoadedState(attached));
    await store.getState().liveAttachTrainingWeights();
    expect(store.getState().error).toBe("Training weight live attach failed");
  });
});

describe("attaching weights to a new run", () => {
  it("asks for a completed job, and attaches nothing while a run is busy", async () => {
    const { store } = studioStore();
    await store.getState().attachTrainingWeights();
    expect(store.getState().error).toBe("No completed training job is available.");
    store.setState({ trainingJobId: "sj_1", trainingStatus: "running" });
    await store.getState().attachTrainingWeights();
    expect(attachTrainingWeights).not.toHaveBeenCalled();
  });

  it.each([
    ["warm_start", { source_job_id: "sj_1", job_id: "sj_2", status: "running" }],
    ["warm_start", { source_job_id: "sj_1", job_id: "sj_2", status: "running", mode: "warm_start" }],
    ["exact_resume", { source_job_id: "sj_1", job_id: "sj_2", status: "running", mode: "exact_resume" }],
  ] as const)("starts the %s run the server attaches and follows it", async (mode, response) => {
    const { store } = studioStore();
    recordStreams();
    store.setState({ trainingJobId: "sj_1", trainingStatus: "completed" });
    vi.mocked(attachTrainingWeights).mockResolvedValue(answer(attachTrainingWeights, response));
    vi.mocked(fetchTrainingJobs).mockResolvedValue(answer(fetchTrainingJobs, []));
    await store.getState().attachTrainingWeights(mode);
    expect(store.getState().trainingJobId).toBe("sj_2");
    expect(store.getState().trainingStatus).toBe("running");
    expect(latestStream().jobId).toBe("sj_2");
  });

  it.each([
    ["warm_start", { source_job_id: "sj_other", job_id: "sj_2", status: "running" }],
    ["warm_start", { source_job_id: "sj_1", job_id: 7, status: "running" }],
    ["warm_start", { source_job_id: "sj_1", job_id: "", status: "running" }],
    ["warm_start", { source_job_id: "sj_1", job_id: "sj_2", status: "queued" }],
    ["warm_start", { source_job_id: "sj_1", job_id: "sj_2", status: "running", mode: "exact_resume" }],
    ["exact_resume", { source_job_id: "sj_1", job_id: "sj_2", status: "running" }],
  ] as const)("refuses an attachment (%s) the server did not make as asked, and restores the status", async (mode, response) => {
    const { store } = studioStore();
    store.setState({ trainingJobId: "sj_1", trainingStatus: "completed" });
    vi.mocked(attachTrainingWeights).mockResolvedValue(answer(attachTrainingWeights, response));
    await store.getState().attachTrainingWeights(mode);
    expect(store.getState().error).toBe("Training attachment response does not identify the requested operation");
    expect(store.getState().trainingStatus).toBe("completed");
  });

  it("only lists the jobs when the attachment was outdated, and drops an outdated refusal", async () => {
    const { store, release } = studioStore();
    store.setState({ trainingJobId: "sj_1", trainingStatus: "completed" });
    vi.mocked(fetchTrainingJobs).mockResolvedValue(answer(fetchTrainingJobs, []));
    const answered = deferred<Awaited<ReturnType<typeof attachTrainingWeights>>>();
    vi.mocked(attachTrainingWeights).mockReturnValueOnce(answered.promise);
    const attach = store.getState().attachTrainingWeights();
    release();
    answered.resolve(answer(attachTrainingWeights, { source_job_id: "sj_1", job_id: "sj_2", status: "running" }));
    await attach;
    expect(fetchTrainingJobs).toHaveBeenCalled();
    expect(store.getState().trainingJobId).toBe("sj_1");
    store.setState({ trainingStatus: "completed" });
    const refused = deferred<Awaited<ReturnType<typeof attachTrainingWeights>>>();
    vi.mocked(attachTrainingWeights).mockReturnValueOnce(refused.promise);
    const late = store.getState().attachTrainingWeights();
    release();
    refused.reject(new Error("late"));
    await late;
    expect(store.getState().error).toBeNull();
  });
});

describe("the configuration and the release hook", () => {
  it("updates one field of the training configuration", () => {
    const { store } = studioStore();
    store.getState().setTrainingConfig("epochs", 7);
    expect(store.getState().trainingConfig.epochs).toBe(7);
  });

  it("signing out releases the training the store follows", async () => {
    recordStreams();
    const store = createStore<StudioState>((set, get) => ({
      ...studioInitialData, ...createStudioStoreActions(set, get), ...createStudioEquationIrActions(set, get),
    }));
    store.setState({ trainingJobs: [job("sj_1", "running")] as StudioState["trainingJobs"] });
    vi.mocked(fetchTrainingStatus).mockResolvedValueOnce(status("sj_1", "running"));
    await store.getState().selectTrainingJob("sj_1");
    vi.mocked(logoutStudioBrowserUser).mockResolvedValue(undefined as never);
    await store.getState().logoutBrowserUser();
    expect(latestStream().close).toHaveBeenCalledOnce();
    expect(store.getState().trainingJobId).toBeNull();
  });

  it("closes the followed stream on release and drops every reply still on its way", async () => {
    const { store, release } = await following();
    const stream = latestStream();
    const reply = deferred<Awaited<ReturnType<typeof fetchTrainingJobs>>>();
    vi.mocked(fetchTrainingJobs).mockReturnValueOnce(reply.promise);
    const listing = store.getState().loadTrainingJobs();
    release();
    expect(stream.close).toHaveBeenCalledOnce();
    reply.resolve(answer(fetchTrainingJobs, [job("sj_late", "running")]));
    await listing;
    expect(store.getState().trainingJobs.map((row) => row.job_id)).toEqual(["sj_1"]);
    stream.handlers.onEpoch({ epoch: 9 } as unknown as Parameters<StreamHandlers["onEpoch"]>[0]);
    expect(store.getState().trainingEpochs).toEqual([]);
  });
});
