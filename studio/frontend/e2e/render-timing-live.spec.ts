// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — how long the built Studio takes to show a result, against the real backend

/**
 * Measure, ten times each, how long the built Studio takes to draw a result.
 *
 * Each sample puts a start mark on the page's timeline just before a user
 * action, then waits for the first render timing mark (`src/renderTiming.ts`)
 * after it; the sample is the time between the two. The workloads are the
 * shipped catalogue, one model at about 1 000, 10 000 and 100 000 steps,
 * networks of 2, 10 and 50 populations in canvas and table view, and a
 * 50-epoch training stream on the synthetic dataset. The run writes
 * one `studio.render-timing.v1` artefact (`e2e/renderTimingReport.ts`) to
 * `SC_NEUROCORE_RENDER_TIMING_OUT`, or to the test's output folder.
 *
 * The numbers describe this code on this host at its load. Unless the run was
 * given reserved cores (`SC_NEUROCORE_TIMING_RESERVED_CORES`), the artefact is
 * labelled `functional-non-isolated`. No budget is checked here.
 */
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { cpus, hostname, loadavg } from "node:os";
import { dirname, join } from "node:path";

import { expect, test, type Page } from "@playwright/test";

import { MIN_SAMPLES, buildArtefact, commitLabel, isolationLabel, type MeasureSamples } from "./renderTimingReport";

test.describe.configure({ mode: "serial" });

const SAMPLES = MIN_SAMPLES;
const START = "studio:measure-start";
/** The states in which a training run has ended, whether or not it completed. */
const TRAINING_ENDED = /^(completed|failed|stopped|interrupted)$/;
/**
 * How long a sample waits for its end mark, in milliseconds
 * (`SC_NEUROCORE_RENDER_TIMING_MARK_TIMEOUT_MS`, default five minutes). A mark
 * that never comes fails the run: a measure without its end is not measured.
 */
const MARK_TIMEOUT = Number(process.env.SC_NEUROCORE_RENDER_TIMING_MARK_TIMEOUT_MS ?? "300000");
const measures: MeasureSamples[] = [];
let loadBefore: number[] = [];
let browserVersion = "";

/**
 * Return the commit the measured bundle was built from.
 *
 * @returns `GITHUB_SHA` on a hosted run; otherwise the checkout's HEAD, marked
 *   when a tracked file differs from it (`commitLabel`).
 */
function commit(): string {
  if (process.env.GITHUB_SHA !== undefined) return process.env.GITHUB_SHA;
  const git = (...args: string[]): string => execFileSync("git", args, { encoding: "utf8" });
  return commitLabel(git("rev-parse", "HEAD").trim(), git("status", "--porcelain", "--untracked-files=no"));
}

/**
 * Read the duration a waited-for function returned.
 *
 * @param handle - What `waitForFunction` resolved with: a time, never null once it resolves.
 * @returns The milliseconds.
 */
async function milliseconds(handle: { jsonValue(): Promise<number | null> }): Promise<number> {
  const value = await handle.jsonValue();
  if (value === null) throw new Error("the waited-for mark has no time");
  return value;
}

/**
 * Take one sample: mark the start, act, and wait for the first mark of a name after the start.
 *
 * @param page - The page.
 * @param mark - The render timing mark that ends the sample (without its prefix).
 * @param action - The user action.
 * @returns The milliseconds from the start mark to that mark.
 */
async function sample(page: Page, mark: string, action: () => Promise<void>): Promise<number> {
  await page.evaluate((name) => {
    performance.clearMarks(name);
    performance.mark(name);
  }, START);
  await action();
  const handle = await page.waitForFunction(
    ({ start, end }) => {
      const begun = performance.getEntriesByName(start, "mark")[0];
      if (begun === undefined) return null;
      const ended = performance.getEntriesByName(end, "mark").find((entry) => entry.startTime >= begun.startTime);
      return ended === undefined ? null : ended.startTime - begun.startTime;
    },
    { start: START, end: `studio:${mark}` },
    { timeout: MARK_TIMEOUT },
  );
  return milliseconds(handle);
}

/**
 * Record one measure's samples.
 *
 * @param name - The measure.
 * @param workload - What it ran on.
 * @param samplesMs - Its samples.
 */
function record(name: string, workload: string, samplesMs: number[]): void {
  expect(samplesMs).toHaveLength(SAMPLES);
  measures.push({ name, workload, samplesMs });
}

/**
 * Open the Studio with the onboarding dismissed.
 *
 * @param page - The page.
 */
async function open(page: Page): Promise<void> {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
  });
  await page.goto("./");
}

/**
 * Choose a model in the catalogue and wait until its trace is drawn.
 *
 * @param page - The page.
 * @param name - The model.
 */
async function chooseModel(page: Page, name: string): Promise<void> {
  await page.getByPlaceholder("Search models...").fill(name);
  await page.getByTestId(`model-contract-${name}`).locator("..").click();
  await expect(page.getByRole("img", { name: /^Trace of / })).toBeVisible({ timeout: 300_000 });
}

test.beforeAll(({ browser }) => {
  loadBefore = loadavg();
  browserVersion = `${browser.browserType().name()} ${browser.version()}`;
});

test.afterAll(() => {
  // A run whose every workload failed has nothing to record; its failures are reported already.
  if (measures.length === 0) return;
  const artefact = buildArtefact(
    measures,
    {
      host: hostname(),
      cpuModel: cpus()[0]?.model ?? "unknown",
      loadBefore,
      loadAfter: loadavg(),
      browser: browserVersion,
      commit: commit(),
      isolation: isolationLabel(process.env.SC_NEUROCORE_TIMING_RESERVED_CORES),
    },
    new Date().toISOString().replace(/\.\d{3}Z$/, "Z"),
  );
  const out = process.env.SC_NEUROCORE_RENDER_TIMING_OUT ?? join(test.info().project.outputDir, "render-timing.json");
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, `${JSON.stringify(artefact, null, 2)}\n`);
});

test("the shipped catalogue is drawn after loading the page", async ({ page }) => {
  const samples: number[] = [];
  for (let index = 0; index < SAMPLES; index += 1) {
    await open(page);
    const drawn = await page.waitForFunction(
      () => performance.getEntriesByName("studio:catalogue-drawn", "mark")[0]?.startTime ?? null,
      undefined,
      { timeout: MARK_TIMEOUT },
    );
    samples.push(await milliseconds(drawn));
  }
  record("catalogue first draw (from navigation)", "shipped catalogue", samples);
});

test("a simulation request is drawn as a trace, at three run lengths", async ({ page }) => {
  await open(page);
  await chooseModel(page, "AdExNeuron");
  for (const [dt, steps] of [["1", "1 000"], ["0.1", "10 000"], ["0.01", "100 000"]] as const) {
    await page.getByTestId("slider-dt").fill(dt);
    await page.getByTestId("slider-T (ms)").fill("1000");
    await expect(page.getByRole("img", { name: /^Trace of / })).toBeVisible({ timeout: 300_000 });
    const samples: number[] = [];
    for (let index = 0; index < SAMPLES; index += 1) {
      // Alternating 990 and 1 000 ms makes every sample a new request.
      const duration = index % 2 === 0 ? "990" : "1000";
      samples.push(await sample(page, "trace-drawn", () => page.getByTestId("slider-T (ms)").fill(duration)));
    }
    record("simulation request to trace draw", `AdExNeuron, dt ${dt} ms, about ${steps} steps (990 or 1 000 ms)`, samples);
  }
});

test("a model switch is drawn as the new model's trace", async ({ page }) => {
  await open(page);
  await chooseModel(page, "AdExNeuron");
  const samples: number[] = [];
  for (let index = 0; index < SAMPLES; index += 1) {
    const next = index % 2 === 0 ? "Izhikevich2007Neuron" : "AdExNeuron";
    samples.push(await sample(page, "trace-drawn", () => chooseModel(page, next)));
  }
  record("model switch to trace draw", "AdExNeuron and Izhikevich2007Neuron, default run", samples);
});

test("a view switch is drawn", async ({ page }) => {
  await open(page);
  await chooseModel(page, "AdExNeuron");
  const samples: number[] = [];
  for (let index = 0; index < SAMPLES; index += 1) {
    const view = index % 2 === 0 ? "Phase plane" : "Trace";
    samples.push(await sample(page, "view-drawn", () => page.getByRole("tab", { name: view, exact: true }).click()));
  }
  record("view switch to draw", "AdExNeuron, Trace and Phase plane", samples);
});

for (const size of [2, 10, 50]) {
  test(`a network of ${String(size)} populations is redrawn in canvas and table view`, async ({ page }) => {
    await open(page);
    const models = page.waitForResponse(
      (response) => new URL(response.url()).pathname === "/api/graph/models" && response.ok(),
    );
    await page.getByRole("tab", { name: "Network canvas" }).click();
    await models;
    for (let index = 0; index < size; index += 1) {
      await page.getByRole("button", { name: index % 2 === 0 ? "+ Exc" : "+ Inh", exact: true }).click();
    }
    await expect(page.locator(".react-flow__node")).toHaveCount(size);

    // New populations go in a row 200 px apart, so a large network does not fit
    // the view at its smallest zoom: drag the first population whose centre is
    // on screen and is the population itself, not a neighbour or the page.
    const grabbed = await page.evaluate(() => {
      for (const element of document.querySelectorAll<HTMLElement>(".react-flow__node")) {
        const box = element.getBoundingClientRect();
        const centre = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
        if (document.elementFromPoint(centre.x, centre.y)?.closest(".react-flow__node") === element) {
          return { id: element.dataset.id ?? "", ...centre };
        }
      }
      return null;
    });
    if (grabbed === null) throw new Error("no population can be grabbed on the canvas");
    const node = page.locator(`.react-flow__node[data-id="${grabbed.id}"]`);
    const { x, y } = grabbed;
    await page.mouse.move(x, y);
    await page.mouse.down();
    // The first move only starts the drag (React Flow marks the node as
    // dragging); positions change from the next move on, so the samples start there.
    await page.mouse.move(x + 4, y, { steps: 1 });
    await expect(node).toHaveClass(/\bdragging\b/);
    const canvas: number[] = [];
    for (let index = 0; index < SAMPLES; index += 1) {
      canvas.push(await sample(page, "network-drawn", () => page.mouse.move(x + 8 * (index + 1), y, { steps: 1 })));
    }
    await page.mouse.up();
    record("network canvas drag frame to redraw", `${String(size)} populations, canvas view`, canvas);

    const table: number[] = [];
    for (let index = 0; index < SAMPLES; index += 1) {
      table.push(await sample(page, "network-drawn", () => page.getByRole("button", { name: "Table view", exact: true }).click()));
    }
    record("network canvas and table switch to redraw", `${String(size)} populations, canvas and table view`, table);
  });
}

test("each training epoch's event is drawn in the charts", async ({ page }) => {
  await open(page);
  await page.getByRole("tab", { name: "Training" }).click();
  await page.getByLabel("Epochs", { exact: true }).fill("50");
  await page.getByLabel("Batch Size", { exact: true }).fill("32");
  await page.getByLabel("Timesteps", { exact: true }).fill("4");
  const started = page.waitForRequest((request) => new URL(request.url()).pathname === "/api/training/start");
  await page.getByRole("button", { name: "Train", exact: true }).last().click();
  const start = (await (await (await started).response())?.json()) as { job_id: string };
  let status = "";
  await expect.poll(async () => {
    const answer = await page.request.get(`/api/training/status/${start.job_id}`);
    status = ((await answer.json()) as { status: string }).status;
    return status;
  }, { timeout: 1_200_000 }).toMatch(TRAINING_ENDED);
  // A run that ended any other way fails here, with its status, not after the whole wait.
  expect(status).toBe("completed");
  // Each epoch's arrival is paired with the first chart draw after it.
  const samples = await page.evaluate(() => {
    const draws = performance.getEntriesByName("studio:training-chart-drawn", "mark").map((entry) => entry.startTime);
    return performance.getEntriesByName("studio:training-event-received", "mark").flatMap((event) => {
      const drawn = draws.find((time) => time >= event.startTime);
      return drawn === undefined ? [] : [drawn - event.startTime];
    });
  });
  expect(samples.length).toBeGreaterThanOrEqual(SAMPLES);
  measures.push({ name: "training epoch event to chart draw", workload: "synthetic dataset, 50 epochs, batch 32, 4 timesteps", samplesMs: samples });
});
