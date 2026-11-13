// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — the experiment cohort workbench, mounted with stand-ins for its job and its requests
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { CohortResult, MeasuredPareto } from "../api/fitsApi";
import CohortPanel from "./CohortPanel";

const job = vi.hoisted(() => ({
  id: null as string | null,
  status: "idle",
  result: null as CohortResult | null,
  error: null as string | null,
  busy: false,
  start: vi.fn(async (submit: () => Promise<unknown>) => { await submit(); }),
  cancel: vi.fn(async () => undefined),
}));
const api = vi.hoisted(() => ({
  submitCohort: vi.fn(async () => ({ job_id: "cohort-1", status_route: "/api/jobs/cohort-1" })),
  replayCohort: vi.fn(),
  compareMeasurements: vi.fn(),
  download: vi.fn(),
}));

vi.mock("../useLaboratoryJob", () => ({ useLaboratoryJob: () => job }));
vi.mock("../api/fitsApi", () => ({
  submitCohort: api.submitCohort,
  replayCohort: api.replayCohort,
  compareMeasurements: api.compareMeasurements,
}));
vi.mock("../browserArtefactDownload", () => ({ downloadBrowserArtefact: api.download }));

const RESULT: CohortResult = {
  schema_version: "sc-neurocore.cohort-result.v1",
  cohort: {},
  trials: [
    {
      model: "LIFNeuron", parameters: { tau: 10, gain: 2 }, status: "completed", rejected_constraints: [],
      metric: { kind: "rate_error", unit: "Hz", observable: "rate" },
      samples: [{ sample: "s1", split: "train", value: 1.234567, failed: false }, { sample: "s2", split: "test", value: null, failed: true }],
      trial_sha256: "a1",
    },
    {
      model: "AdExNeuron", parameters: {}, status: "rejected", rejected_constraints: ["budget", "units"],
      metric: { kind: "rate_error", unit: "Hz", observable: "rate" }, samples: [], trial_sha256: "b2",
    },
  ],
  selection: [{ model: "LIFNeuron", trial_sha256: "a1", training_metric: 0.123456 }, { model: "AdExNeuron", trial_sha256: null, reason: "rejected" }],
  measurement_status: "No measurement receipts supplied.",
  result_sha256: "f0",
};

let root: Root | undefined;
let container: HTMLDivElement;

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  Object.assign(job, { id: null, status: "idle", result: null, error: null, busy: false });
  job.start.mockClear();
  job.cancel.mockClear();
  for (const mock of [api.submitCohort, api.replayCohort, api.compareMeasurements, api.download]) mock.mockClear();
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root?.unmount(); await Promise.resolve(); });
  document.body.replaceChildren();
  vi.unstubAllGlobals();
});

/** Mount the workbench. */
async function mount(): Promise<void> {
  await act(async () => { root?.render(<CohortPanel />); await Promise.resolve(); });
}

/**
 * The button with this text.
 *
 * @param text - Its label.
 * @returns The button.
 */
function button(text: string): HTMLButtonElement {
  const found = [...container.querySelectorAll("button")].find((item) => item.textContent === text);
  if (found === undefined) throw new Error(`no button reads ${text}`);
  return found;
}

/**
 * Give a file input a file, as a person picking one would.
 *
 * @param id - The input's id.
 * @param file - The file, or an object standing in for one.
 */
async function pick(id: string, file: File): Promise<void> {
  const input = container.querySelector<HTMLInputElement>(`#${id}`);
  if (input === null) throw new Error(`no input #${id}`);
  Object.defineProperty(input, "files", { configurable: true, value: [file] });
  await act(async () => {
    input.dispatchEvent(new Event("change", { bubbles: true }));
    for (let turn = 0; turn < 5; turn += 1) await Promise.resolve();
  });
}

/** The status region's text. */
function status(): string {
  return container.querySelector('[role="status"][aria-label="Cohort status"]')?.textContent ?? "";
}

/**
 * An error from another realm, such as a worker or a frame: shaped and printed
 * like an Error, but not an instance of this realm's Error.
 */
class ForeignRealmError {
  readonly name = "NotReadableError";
  readonly message: string;

  /**
   * Make the error.
   *
   * @param message - What went wrong.
   */
  constructor(message: string) {
    this.message = message;
  }

  /**
   * Print the error.
   *
   * @returns The name and the message, as Error prints them.
   */
  toString(): string {
    return `${this.name}: ${this.message}`;
  }
}

/**
 * A file whose text cannot be read, failing with an error from another realm.
 *
 * @param name - The file's name.
 * @returns The file.
 */
function unreadable(name: string): File {
  const file = new File(["{}"], name);
  const failure: Error = new ForeignRealmError("the disk went away");
  Object.defineProperty(file, "text", { value: () => Promise.reject(failure) });
  return file;
}

describe("CohortPanel", () => {
  it("cannot run before a cohort is imported, and shows no result", async () => {
    await mount();
    expect(button("Run cohort").disabled).toBe(true);
    expect(container.querySelector("table")).toBeNull();
    expect(status()).toBe("");
  });

  it("imports a cohort document, then submits exactly that document", async () => {
    await mount();
    await pick("cohort-import", new File([JSON.stringify({ trials: 3 })], "cohort.json"));
    expect(status()).toBe("Imported cohort.json; execution validates units, splits and budgets.");
    expect(button("Run cohort").disabled).toBe(false);
    await act(async () => { button("Run cohort").click(); await Promise.resolve(); });
    expect(job.start).toHaveBeenCalledTimes(1);
    expect(api.submitCohort).toHaveBeenCalledWith({ trials: 3 });
  });

  it("states why a document could not be imported", async () => {
    await mount();
    await pick("cohort-import", new File(["{not json"], "broken.json"));
    expect(status()).toMatch(/JSON/);
    expect(button("Run cohort").disabled).toBe(true);
    await pick("cohort-import", unreadable("gone.json"));
    expect(status()).toBe("NotReadableError: the disk went away");
  });

  it("while a cohort runs, refuses another run, offers to cancel, and reports the job", async () => {
    Object.assign(job, { busy: true, id: "cohort-1", status: "running", error: "worker restarted" });
    await mount();
    expect(button("Run cohort").disabled).toBe(true);
    expect(status()).toContain("Background cohort: running (cohort-1)");
    expect(status()).toContain("worker restarted");
    await act(async () => { button("Cancel cohort").click(); await Promise.resolve(); });
    expect(job.cancel).toHaveBeenCalledTimes(1);
  });

  it("tables every trial with its parameters, rejected constraints and samples, and the training-only selection", async () => {
    job.result = RESULT;
    await mount();
    const rows = [...container.querySelectorAll("table tbody tr")].map((row) => [...row.children].map((cell) => cell.textContent));
    expect(rows).toEqual([
      ["LIFNeuron", "tau=10, gain=2", "rate_error / Hz", "completed", "s1: train 1.2346; s2: test failed"],
      ["AdExNeuron", "", "rate_error / Hz", "rejected (budget, units)", ""],
    ]);
    expect(container.querySelector("caption")?.textContent).toBe("Complete sweep trials");
    expect([...container.querySelectorAll('ul[aria-label="Training-only model selection"] li')].map((item) => item.textContent)).toEqual([
      "LIFNeuron: training metric 0.12346",
      "AdExNeuron: rejected",
    ]);
    expect(container.textContent).toContain("No measurement receipts supplied.");
  });

  it("exports the full result as one JSON file", async () => {
    job.result = RESULT;
    await mount();
    await act(async () => { button("Export full cohort result").click(); await Promise.resolve(); });
    expect(api.download).toHaveBeenCalledTimes(1);
    const [blob, name] = api.download.mock.calls[0] as [Blob, string];
    expect(name).toBe("cohort-result.json");
    expect(JSON.parse(await blob.text())).toEqual(RESULT);
  });

  it("replays the cohort and says whether it reproduced, refusing a second replay meanwhile", async () => {
    job.result = RESULT;
    let finish: (value: { reproduced: boolean }) => void = () => undefined;
    api.replayCohort.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
    await mount();
    await act(async () => { button("Replay full cohort").click(); await Promise.resolve(); });
    expect(api.replayCohort).toHaveBeenCalledWith(RESULT);
    expect(button("Replay full cohort").disabled).toBe(true);
    await act(async () => { finish({ reproduced: true }); for (let turn = 0; turn < 5; turn += 1) await Promise.resolve(); });
    expect(status()).toBe("Cohort replay reproduced the full result.");
    expect(button("Replay full cohort").disabled).toBe(false);
    api.replayCohort.mockResolvedValueOnce({ reproduced: false });
    await act(async () => { button("Replay full cohort").click(); for (let turn = 0; turn < 5; turn += 1) await Promise.resolve(); });
    expect(status()).toBe("Cohort replay did not reproduce the full result.");
  });

  it("states why a replay failed, whatever was thrown", async () => {
    job.result = RESULT;
    await mount();
    api.replayCohort.mockRejectedValueOnce(new Error("replay refused"));
    await act(async () => { button("Replay full cohort").click(); for (let turn = 0; turn < 5; turn += 1) await Promise.resolve(); });
    expect(status()).toBe("replay refused");
    api.replayCohort.mockRejectedValueOnce("network down");
    await act(async () => { button("Replay full cohort").click(); for (let turn = 0; turn < 5; turn += 1) await Promise.resolve(); });
    expect(status()).toBe("network down");
  });

  it("says why measured tradeoffs have no frontier", async () => {
    job.result = RESULT;
    const comparison: MeasuredPareto = { comparable: false, custody: "Receipts from one lab.", reason: "different targets", rows: [] };
    api.compareMeasurements.mockResolvedValueOnce(comparison);
    await mount();
    await pick("measurement-import", new File([JSON.stringify([{ receipt: 1 }])], "receipts.json"));
    expect(api.compareMeasurements).toHaveBeenCalledWith(RESULT, [{ receipt: 1 }]);
    const tradeoffs = container.querySelector('[aria-label="Measured tradeoffs"]');
    expect(tradeoffs?.textContent).toBe("Receipts from one lab.No frontier: different targets");
  });

  it("tables comparable measurements with their units and which are nondominated", async () => {
    job.result = RESULT;
    api.compareMeasurements.mockResolvedValueOnce({
      comparable: true, custody: "Receipts from one rig.", reason: null,
      contract: { resource_unit: "LUT", target: "ice40" }, metric: { unit: "Hz", kind: "rate_error" },
      rows: [
        { model: "LIFNeuron", trial_sha256: "a1", holdout_error: 0.5, latency_ms: 2, resources: 120, energy_j: 0.01, nondominated: true },
        { model: "AdExNeuron", trial_sha256: "b2", holdout_error: 0.7, latency_ms: 3, resources: 300, energy_j: 0.02, nondominated: false },
      ],
    } satisfies MeasuredPareto);
    await mount();
    await pick("measurement-import", new File(["[]"], "receipts.json"));
    const table = container.querySelector('[aria-label="Measured tradeoffs"] table');
    expect([...(table?.querySelectorAll("th[scope=col]") ?? [])].map((cell) => cell.textContent)).toEqual([
      "Model", "Holdout error (Hz)", "Latency (ms)", "Resources (LUT)", "Energy (J)", "Nondominated",
    ]);
    expect([...(table?.querySelectorAll("tbody tr") ?? [])].map((row) => row.lastElementChild?.textContent)).toEqual(["yes", "no"]);
  });

  it("leaves the units blank when the comparison names none", async () => {
    job.result = RESULT;
    api.compareMeasurements.mockResolvedValueOnce({ comparable: true, custody: "", reason: null, rows: [] } satisfies MeasuredPareto);
    await mount();
    await pick("measurement-import", new File(["[]"], "receipts.json"));
    const headings = [...container.querySelectorAll('[aria-label="Measured tradeoffs"] th[scope=col]')].map((cell) => cell.textContent);
    expect(headings).toContain("Holdout error ()");
    expect(headings).toContain("Resources ()");
  });

  it("states why measurement receipts could not be read, whatever was thrown", async () => {
    job.result = RESULT;
    await mount();
    await pick("measurement-import", new File(["{not json"], "receipts.json"));
    expect(status()).toMatch(/JSON/);
    await pick("measurement-import", unreadable("receipts.json"));
    expect(status()).toBe("NotReadableError: the disk went away");
  });

  it("ignores either file input when it was cleared without a file", async () => {
    job.result = RESULT;
    await mount();
    for (const id of ["cohort-import", "measurement-import"]) {
      const input = container.querySelector<HTMLInputElement>(`#${id}`);
      if (input === null) throw new Error(`no input #${id}`);
      Object.defineProperty(input, "files", { configurable: true, value: [] });
      await act(async () => { input.dispatchEvent(new Event("change", { bubbles: true })); await Promise.resolve(); });
    }
    expect(status()).toBe("");
    expect(api.compareMeasurements).not.toHaveBeenCalled();
  });
});
