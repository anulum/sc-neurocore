// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Live hardware catalogue discovery and workbench selection

import { expect, test } from "@playwright/test";
import type { HardwareCatalogueResponse } from "../src/hardwareCatalogueTypes";

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("sc-studio-onboarding-dismissed", "true");
  });
});

test("actual registration discovery preserves all targets and selects without executing tools", async ({ page, browser }) => {
  const operations: string[] = [];
  page.on("request", (request) => {
    if (request.method() === "POST") operations.push(new URL(request.url()).pathname);
  });
  const entryResponse = await page.goto("./");
  if (entryResponse === null) throw new Error("Studio entry returned no HTTP response");
  await test.info().attach("served-ui-index.html", { body: await entryResponse.body(), contentType: "text/html" });
  await test.info().attach("browser-version", { body: browser.version(), contentType: "text/plain" });
  await page.getByRole("button", { name: "Models", exact: true }).click();
  const response = page.waitForResponse((item) =>
    new URL(item.url()).pathname === "/api/hardware/catalogue" && item.ok(),
  );
  await page.getByRole("tab", { name: "FPGA synthesis", exact: true }).click();
  const catalogue = (await (await response).json()) as HardwareCatalogueResponse;
  await test.info().attach("hardware-catalogue.json", {
    body: JSON.stringify(catalogue, null, 2), contentType: "application/json",
  });
  const section = page.getByRole("region", { name: "Hardware catalogue", exact: true });
  await section.getByText("Hardware catalogue", { exact: true }).click();
  await expect(section.getByRole("article")).toHaveCount(catalogue.entries.length);
  for (const entry of catalogue.entries) {
    await expect(section.getByRole("article", { name: entry.name, exact: true })).toBeVisible();
    expect(entry.evidence_level).toBe("registration");
    expect(entry.physical_evidence).toEqual([]);
  }
  expect(catalogue.entries.map((entry) => entry.target_id)).toEqual(expect.arrayContaining([
    "brainscales2", "brainscales3", "dynap_se", "loihi2", "spinnaker2", "board:pynq_z2",
    "fpga:ice40", "fpga:ecp5", "fpga:gowin", "fpga:xilinx",
  ]));
  const target = page.getByRole("combobox", { name: "FPGA target", exact: true });
  const profiles = catalogue.entries.flatMap((entry) =>
    entry.synthesis_profile === null ? [] : [entry.synthesis_profile.target],
  );
  await expect(target.locator("option")).toHaveCount(profiles.length);
  expect(await target.locator("option").evaluateAll((options) =>
    options.map((option) => (option as HTMLOptionElement).value),
  )).toEqual(profiles);
  await expect(section.getByRole("button", { name: "Use GOWIN target", exact: true })).toBeDisabled();
  await section.getByRole("button", { name: "Use ECP5 target", exact: true }).click();
  await expect(target).toHaveValue("ecp5");

  await section.getByRole("textbox", { name: "Search hardware" }).fill(" DyNaP ");
  await expect(section.getByRole("article")).toHaveCount(1);
  await expect(section.getByRole("article", { name: "DYNAP-SE" })).toContainText("capability_manifest");
  await expect(target).toHaveValue("ecp5");
  await section.getByRole("textbox", { name: "Search hardware" }).fill("");
  await section.getByRole("combobox", { name: "Hardware category" }).selectOption("board");
  await expect(section.getByRole("article")).toHaveCount(1);
  await expect(section.getByRole("article", { name: "PYNQ-Z2" })).toContainText("EMULATION mode");
  await page.screenshot({ path: test.info().outputPath("hardware-catalogue.png") });

  await page.getByRole("button", { name: "ODE", exact: true }).click();
  await page.getByRole("tab", { name: "FPGA synthesis", exact: true }).click();
  const odeCatalogue = page.getByRole("region", { name: "Hardware catalogue", exact: true });
  if (await odeCatalogue.locator("details").getAttribute("open") === null) {
    await odeCatalogue.getByText("Hardware catalogue", { exact: true }).click();
  }
  await odeCatalogue.getByRole("combobox", { name: "Hardware category" }).selectOption("fpga");
  await odeCatalogue.getByRole("button", { name: "Use GOWIN target", exact: true }).click();
  await expect(page.getByRole("combobox", { name: "FPGA target", exact: true })).toHaveValue("gowin");
  expect(operations.filter((path) =>
    path.startsWith("/api/synth/") || path.startsWith("/api/deploy")
    || path.includes("/jobs"),
  )).toEqual([]);
});

test("an actual network refusal stays visible and retry reads the real backend", async ({ page }) => {
  await page.route("**/api/hardware/catalogue", async (route) => { await route.abort("failed"); });
  await page.goto("./");
  await page.getByRole("tab", { name: "FPGA synthesis", exact: true }).click();
  const section = page.getByRole("region", { name: "Hardware catalogue", exact: true });
  await section.getByText("Hardware catalogue", { exact: true }).click();
  await expect(section.getByRole("alert")).toBeVisible();
  await expect(section.getByRole("article")).toHaveCount(0);
  await page.unroute("**/api/hardware/catalogue");
  const response = page.waitForResponse((item) =>
    new URL(item.url()).pathname === "/api/hardware/catalogue" && item.ok(),
  );
  await section.getByRole("button", { name: "Retry hardware catalogue" }).click();
  const catalogue = (await (await response).json()) as HardwareCatalogueResponse;
  await expect(section.getByRole("article")).toHaveCount(catalogue.entries.length);
});

test("toolbar discovery works independently of optional synthesis tools", async ({ page, browser }) => {
  const operations: string[] = [];
  page.on("request", (request) => {
    if (request.method() === "POST") operations.push(new URL(request.url()).pathname);
  });
  const initialSimulation = page.waitForResponse((response) =>
    new URL(response.url()).pathname === "/api/models/simulate" && response.ok(),
  );
  const entry = await page.goto("./");
  if (entry === null) throw new Error("Studio entry returned no HTTP response");
  await test.info().attach("served-ui-index.html", { body: await entry.body(), contentType: "text/html" });
  await test.info().attach("browser-version", { body: browser.version(), contentType: "text/plain" });
  const capabilities = await page.request.get("/api/studio/capabilities");
  expect(capabilities.ok()).toBe(true);
  await test.info().attach("actual-capabilities.json", {
    body: await capabilities.body(), contentType: "application/json",
  });
  await initialSimulation;
  await expect(page.getByRole("button", { name: "Run", exact: true })).toBeVisible();
  operations.length = 0;
  const trigger = page.getByRole("button", { name: "Hardware catalogue", exact: true });
  await expect(trigger).toBeEnabled();
  const response = page.waitForResponse((item) =>
    new URL(item.url()).pathname === "/api/hardware/catalogue" && item.ok(),
  );
  await trigger.focus();
  await page.keyboard.press("Space");
  const catalogue = (await (await response).json()) as HardwareCatalogueResponse;
  await test.info().attach("hardware-catalogue.json", {
    body: JSON.stringify(catalogue, null, 2), contentType: "application/json",
  });
  const modal = page.getByRole("dialog", { name: "Hardware catalogue", exact: true });
  await expect(modal).toBeVisible();
  await expect(modal.getByRole("article")).toHaveCount(catalogue.entries.length);
  await modal.getByRole("textbox", { name: "Search hardware" }).fill("ecp5");
  await modal.getByRole("button", { name: "Use ECP5 target", exact: true }).click();
  await expect(modal.getByRole("button", { name: "Use ECP5 target", exact: true })).toHaveAttribute("aria-pressed", "true");
  await page.keyboard.press("?");
  await expect(page.getByRole("dialog", { name: "Keyboard shortcuts", exact: true })).toHaveCount(0);
  await page.keyboard.press("Escape");
  await expect(modal).toHaveCount(0);
  await expect(trigger).toBeFocused();
  await trigger.click();
  await expect(modal.getByRole("button", { name: "Use ECP5 target", exact: true })).toHaveAttribute("aria-pressed", "true");
  await modal.getByRole("button", { name: "Close hardware catalogue", exact: true }).click();
  await expect(modal).toHaveCount(0);
  await expect(trigger).toBeFocused();
  expect(operations).toEqual([]);
  const odeSimulation = page.waitForResponse((response) =>
    new URL(response.url()).pathname === "/api/simulate" && response.ok(),
  );
  await page.getByRole("button", { name: "ODE", exact: true }).click();
  await odeSimulation;
  await expect(page.getByRole("button", { name: "Run", exact: true })).toBeVisible();
  operations.length = 0;
  await trigger.click();
  await modal.getByRole("textbox", { name: "Search hardware" }).fill("gowin");
  await modal.getByRole("button", { name: "Use GOWIN target", exact: true }).click();
  await expect(modal.getByRole("button", { name: "Use GOWIN target", exact: true })).toHaveAttribute("aria-pressed", "true");
  await page.screenshot({ path: test.info().outputPath("hardware-catalogue-dialog.png") });
  await page.keyboard.press("Escape");
  expect(operations).toEqual([]);
});
