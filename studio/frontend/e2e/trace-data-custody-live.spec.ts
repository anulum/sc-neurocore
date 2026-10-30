// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Long trace custody through the real browser workbench

/** Real catalogue data through display reduction, interaction and downloads. */

import { readFileSync } from "node:fs";
import { expect, test, type Page } from "@playwright/test";
import type { SimulateResponse } from "../src/api/types";
import { parseStudioJson } from "../src/studioJson";

/**
 * Download an export through its actual toolbar action.
 *
 * @param page - The researcher's browser.
 * @param name - The export action.
 * @returns The downloaded UTF-8 content.
 */
async function downloadText(page: Page, name: "JSON" | "CSV"): Promise<string> {
  const downloading = page.waitForEvent("download");
  await page.getByRole("button", { name, exact: true }).click();
  const file = await downloading;
  expect(await file.failure()).toBeNull();
  return readFileSync(await file.path(), "utf8");
}

test("a long trace retains raw sample custody through reduction, pointer interaction and exports", async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
  });
  await page.goto("./");
  await page.getByPlaceholder("Search models...").fill("AdExNeuron");
  const detail = page.waitForResponse((response) =>
    new URL(response.url()).pathname === "/api/models/AdExNeuron" && response.ok());
  await page.getByTestId("model-contract-AdExNeuron").locator("..").click();
  await detail;
  await expect(page.getByTestId("slider-tau_w")).toBeVisible();
  await page.getByTestId("slider-dt").fill("0.1");
  await page.getByTestId("protocol-select").selectOption("sine");
  await page.getByTestId("slider-f (Hz)").fill("37");
  const simulated = page.waitForResponse((response) => {
    if (new URL(response.url()).pathname !== "/api/models/simulate" || !response.ok()) return false;
    const input = response.request().postDataJSON() as Record<string, unknown>;
    return input.model_name === "AdExNeuron" && input.duration === 2000 && input.dt === 0.1
      && input.protocol === "sine" && input.frequency_hz === 37;
  });
  await page.getByTestId("slider-T (ms)").fill("2000");
  const result = parseStudioJson(await (await simulated).text()) as SimulateResponse;
  const raw = result.raw;
  const display = result.display;
  if (!raw?.included || raw.states === undefined || raw.drive === undefined || display === undefined) {
    throw new Error("The supported long run did not retain its raw traces and display index");
  }
  const states = raw.states;
  const drive = raw.drive;
  const names = Object.keys(states);
  expect(result.n_steps).toBe(20_000);
  expect(raw.n_steps).toBe(result.n_steps);
  expect(display.method).toBe("bucket-extrema");
  expect(display.point_count).toBeLessThan(result.n_steps);
  expect(display.point_count).toBeLessThanOrEqual(display.max_points);
  expect(display.sample_index).toHaveLength(display.point_count);
  expect(display.sample_index[0]).toBe(0);
  expect(display.sample_index.at(-1)).toBe(result.n_steps - 1);
  expect(result.time).toEqual(display.sample_index.map((step) => (step + 1) * result.dt));
  expect(drive).toHaveLength(result.n_steps);
  expect(result.current_trace).toEqual(display.sample_index.map((step) => drive[step]));
  for (const name of names) {
    const values = states[name];
    if (values === undefined) throw new Error(`Missing raw state ${name}`);
    expect(values).toHaveLength(result.n_steps);
    expect(result.states[name]).toEqual(display.sample_index.map((step) => values[step]));
  }

  const plot = page.getByRole("img", { name: /^Trace of / });
  await expect(plot).toHaveAccessibleName(/20000 steps/);
  await page.getByRole("button", { name: "Data table", exact: true }).click();
  const table = page.getByRole("table", { name: /^Trace data: / });
  for (const name of names) {
    const values = states[name];
    if (values === undefined) throw new Error(`Missing raw state ${name}`);
    const minimum = values.reduce((a, b) => Math.min(a, b));
    const maximum = values.reduce((a, b) => Math.max(a, b));
    const final = values.at(-1);
    if (final === undefined) throw new Error(`Empty raw state ${name}`);
    const row = table.getByRole("rowheader", { name, exact: true }).locator("..");
    expect(await row.getByRole("cell").allTextContents()).toEqual([
      String(display.point_count), ...[minimum, maximum, final].map((value) => Number(value.toPrecision(4)).toString()),
    ]);
  }
  await page.getByRole("button", { name: "Data table", exact: true }).click();

  const bounds = await plot.boundingBox();
  if (bounds === null) throw new Error("The long trace has no browser geometry");
  const width = bounds.width - 52 - 12;
  const x = bounds.x + 52 + width * 0.23;
  const y = bounds.y + 30;
  const tooltip = page.getByText(/^t=.* step=\d+ /);
  /** Read an actual cursor and bind its displayed time and values to the raw step. */
  const readCursor = async (): Promise<string> => {
    await expect(tooltip).toBeVisible();
    const text = await tooltip.innerText();
    const match = / step=(\d+) /.exec(text);
    if (match === null) throw new Error(`The trace cursor has no raw step: ${text}`);
    const step = Number(match[1]);
    const position = display.sample_index.indexOf(step);
    expect(position).toBeGreaterThanOrEqual(0);
    expect(text).toBe(`t=${((step + 1) * result.dt).toFixed(1)} step=${step} ${names.map((name) => {
      const value = states[name]?.[step];
      if (value === undefined) throw new Error(`Missing ${name} at raw step ${step}`);
      return `${name}=${value.toFixed(2)}`;
    }).join(" ")}`);
    return text;
  };
  await page.mouse.move(x, y);
  const originalCursor = await readCursor();
  await page.mouse.move(bounds.x + 52 + width * 0.75, y);
  await page.mouse.wheel(0, -100);
  await page.mouse.move(x, y);
  await expect(tooltip).not.toHaveText(originalCursor);
  const zoomedCursor = await readCursor();
  await page.mouse.move(bounds.x + 52 + width * 0.6, y);
  await page.mouse.down();
  await page.mouse.move(bounds.x + 52 + width * 0.6 + 60, y);
  await page.mouse.up();
  await page.mouse.move(x, y);
  await expect(tooltip).not.toHaveText(zoomedCursor);
  await readCursor();
  await page.mouse.dblclick(x, y);
  await page.mouse.move(x + 1, y);
  await page.mouse.move(x, y);
  await expect(tooltip).toHaveText(originalCursor);

  expect(parseStudioJson(await downloadText(page, "JSON"))).toEqual(result);
  const [header, ...rows] = (await downloadText(page, "CSV")).split("\n");
  expect(header).toBe(["time", ...names, "current"].join(","));
  expect(rows).toHaveLength(result.n_steps);
  const numeric = rows.map((row) => row.split(",").map(Number));
  expect(numeric.every((row) => row.length === names.length + 2 && row.every(Number.isFinite))).toBe(true);
  const errors = numeric.map((row, step) => Math.abs((row[0] ?? NaN) - (step + 1) * result.dt));
  expect(Math.max(...errors)).toBeLessThanOrEqual(0.000050001);
  for (const [column, name] of names.entries()) {
    const values = states[name];
    if (values === undefined) throw new Error(`Missing exported raw state ${name}`);
    expect(Math.max(...numeric.map((row, step) => Math.abs((row[column + 1] ?? NaN) - (values[step] ?? NaN)))))
      .toBeLessThanOrEqual(0.0000005001);
  }
  expect(Math.max(...numeric.map((row, step) => Math.abs((row.at(-1) ?? NaN) - (drive[step] ?? NaN)))))
    .toBeLessThanOrEqual(0.000050001);
});
