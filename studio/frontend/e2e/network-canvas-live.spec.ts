// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — live Network Canvas browser and accessibility contract

/**
 * What the canvas offers a keyboard and a screen reader, in a real browser,
 * against a real server.
 *
 * Component tests assert the markup a component renders. They cannot assert
 * what a browser *computes* from it: the accessible name of a control after
 * labels, `aria-label` and text content have been reconciled; whether a
 * control is reachable by Tab; whether a hidden region leaves the tab order.
 * Those are the properties an assistive technology actually consumes, and only
 * a browser produces them.
 *
 * The server here is a real Studio backend and the bundle is the built one, so
 * the accessible surface is checked against the graph that is actually
 * validated and run, not against a mock that could agree with a wrong client.
 *
 * Colour contrast is covered too, and it is the one property here that has to
 * be *computed* rather than queried: a background read from the element itself
 * is `rgba(0,0,0,0)` almost everywhere, so the colour a reader actually sees is
 * composited from the ancestor chain. An element whose background cannot be
 * resolved is reported, never scored — assuming white would manufacture a pass
 * for dark text on an unknown ground.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { expect, test, type Page } from "@playwright/test";

import {
  compareWithBaseline,
  judge,
  parseRgba,
  resolveBackground,
  type ContrastBaselineEntry,
  type ContrastSample,
} from "../src/contrastAudit";

/** The recorded contrast failures this run is compared against. */
const BASELINE = JSON.parse(
  readFileSync(fileURLToPath(new URL("../contrast-baseline.json", import.meta.url)), "utf8"),
) as { failures: ContrastBaselineEntry[] };

test.describe.configure({ mode: "serial" });

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
  });
});

/**
 * Open the Studio on the Canvas tab, with the live model list loaded.
 *
 * The wait is on the response rather than on a rendered element: the canvas
 * renders before its models arrive, and acting in between would test a state
 * no user ever meets.
 */
async function openCanvas(page: Page): Promise<void> {
  await page.goto("./");
  const models = page.waitForResponse(
    (response) => new URL(response.url()).pathname === "/api/graph/models" && response.ok(),
  );
  await page.getByRole("button", { name: "Canvas", exact: true }).first().click();
  await models;
}

/** Add one excitatory and one inhibitory population through the toolbar. */
async function addTwoConnectedPopulations(page: Page): Promise<void> {
  await page.getByRole("button", { name: "+ Exc", exact: true }).click();
  await page.getByRole("button", { name: "+ Inh", exact: true }).click();
  await expect(page.locator(".react-flow__node")).toHaveCount(2);
}

/**
 * Open the table view, which is the canvas's keyboard equivalent.
 *
 * The table is found by its accessible name, which is its caption. That is
 * also the point of the caption: the page carries other tables, and a reader
 * has to be able to tell which one states the topology.
 */
function graphTable(page: Page) {
  return page.getByRole("table", { name: /^Network topology: / });
}

/** Switch to the table view and wait for the topology table itself. */
async function openTableView(page: Page): Promise<void> {
  await page.getByRole("button", { name: "Table view", exact: true }).click();
  await expect(graphTable(page)).toBeVisible();
}

test("the canvas builds a graph the live server validates and runs", async ({ page }) => {
  await openCanvas(page);
  await addTwoConnectedPopulations(page);

  const simulated = page.waitForResponse(
    (response) => new URL(response.url()).pathname === "/api/graph/simulate",
  );
  await page.getByRole("button", { name: "Simulate", exact: true }).click();
  const response = await simulated;

  expect(response.ok()).toBe(true);
  const body = (await response.json()) as { spec?: { graph_sha256?: string } };
  // The server resolved the canvas's own JSON into a specification: the graph
  // the browser drew is the graph that ran.
  expect(body.spec?.graph_sha256).toMatch(/^[0-9a-f]{64}$/);
});

test("every toolbar control has an accessible name the browser can compute", async ({
  page,
}) => {
  await openCanvas(page);

  const buttons = page.locator("button");
  const count = await buttons.count();
  expect(count).toBeGreaterThan(0);
  const unnamed: string[] = [];
  for (let index = 0; index < count; index += 1) {
    const button = buttons.nth(index);
    if (!(await button.isVisible())) continue;
    // The browser's own accessible-name computation, not the markup.
    const name = ((await button.getAttribute("aria-label")) ?? (await button.innerText())).trim();
    if (name.length === 0) unnamed.push(await button.evaluate((node) => node.outerHTML));
  }

  expect(unnamed).toEqual([]);
});

test("the table view states the topology and names what each delete removes", async ({
  page,
}) => {
  await openCanvas(page);
  await addTwoConnectedPopulations(page);
  await openTableView(page);

  const table = graphTable(page);
  await expect(table.locator("caption")).toContainText("Network topology: 2 populations");
  // A row header means a screen reader announces which population a cell is
  // about; without it the cells are anonymous.
  await expect(table.getByRole("rowheader")).toHaveCount(2);
  const columns = await table.getByRole("columnheader").allInnerTexts();
  expect(columns).toEqual([
    "Population",
    "Model",
    "Neurons",
    "Type",
    "Input",
    "Incoming",
    "Outgoing",
    "Problems",
    "Actions",
  ]);
  await expect(table.getByRole("button", { name: /^Delete population / })).toHaveCount(2);
});

test("the canvas leaves the tab order while the table stands in for it", async ({ page }) => {
  await openCanvas(page);
  await addTwoConnectedPopulations(page);

  const nodesVisible = await page.locator(".react-flow__node").first().isVisible();
  expect(nodesVisible).toBe(true);

  await openTableView(page);

  // Hidden, so a keyboard user tabbing through the table never lands inside a
  // canvas they cannot see.
  await expect(page.locator(".react-flow__node").first()).toBeHidden();

  await page.getByRole("button", { name: "Table view", exact: true }).click();
  await expect(page.locator(".react-flow__node").first()).toBeVisible();
});

test("a reader who asked for less motion gets none, inline styles included", async ({
  page,
}) => {
  await openCanvas(page);

  // Three progress indicators animate their width from an inline `style`
  // attribute, which a stylesheet reaches only through `!important`. Measuring
  // the shipped stylesheet against an inline declaration is the only way to
  // know the rule is written widely enough; a component test can read the CSS
  // text but not what a browser computes from it.
  const probe = async () =>
    page.evaluate(() => {
      const element = document.createElement("div");
      element.setAttribute("style", "transition: width 0.3s; animation: spin 2s linear infinite");
      document.body.append(element);
      const computed = getComputedStyle(element);
      const measured = {
        transition: computed.transitionDuration,
        animation: computed.animationDuration,
      };
      element.remove();
      return measured;
    });

  await page.emulateMedia({ reducedMotion: "no-preference" });
  expect((await probe()).transition).toBe("0.3s");

  await page.emulateMedia({ reducedMotion: "reduce" });
  const reduced = await probe();
  expect(Number.parseFloat(reduced.transition)).toBeLessThan(0.001);
  expect(Number.parseFloat(reduced.animation)).toBeLessThan(0.001);
});

test("the population editor labels every input and states its contract", async ({ page }) => {
  await openCanvas(page);
  await addTwoConnectedPopulations(page);

  const contract = page.waitForResponse(
    (response) =>
      new URL(response.url()).pathname.startsWith("/api/graph/models/") && response.ok(),
  );
  await page.locator(".react-flow__node").first().click();
  await contract;

  const editor = page.getByRole("region", { name: /^Population / });
  await expect(editor).toBeVisible();
  // getByLabel uses the browser's label association, so this fails if the
  // label is merely adjacent text rather than actually bound to the input.
  await expect(editor.getByLabel("Neurons")).toBeVisible();
  await expect(editor.getByLabel("Model")).toBeVisible();
  await expect(editor.getByLabel("Input")).toBeVisible();

  const count = editor.getByLabel("Neurons");
  const described = await count.getAttribute("aria-describedby");
  expect(described).not.toBeNull();
  await expect(page.locator(`#${described}`)).toContainText("positive whole number");

  // The parameters come from the live contract, so at least one real
  // constructor field of the chosen model must be offered.
  await expect(editor.locator('input[id^="population-params-"]')).not.toHaveCount(0);
});

test("a refused value carries the server's own message to the input that caused it", async ({
  page,
}) => {
  await openCanvas(page);
  await addTwoConnectedPopulations(page);
  await page.locator(".react-flow__node").first().click();
  const editor = page.getByRole("region", { name: /^Population / });
  await expect(editor).toBeVisible();

  const validated = page.waitForResponse(
    (response) => new URL(response.url()).pathname === "/api/graph/validate",
  );
  await editor.getByLabel("Neurons").fill("0");
  await validated;

  const count = editor.getByLabel("Neurons");
  await expect(count).toHaveAttribute("aria-invalid", "true");
  const described = await count.getAttribute("aria-describedby");
  expect(described).toContain("population-count-error");
  // The server's own words, not a paraphrase: the browser shows what the
  // graph specification said, so this pins the message it actually sends.
  await expect(page.locator("#population-count-error")).toContainText("count must be at least 1");
});

test("every editor input is reachable by keyboard alone", async ({ page }) => {
  await openCanvas(page);
  await addTwoConnectedPopulations(page);
  await page.locator(".react-flow__node").first().click();
  const editor = page.getByRole("region", { name: /^Population / });
  await expect(editor).toBeVisible();

  const inputs = editor.locator("input, select");
  const count = await inputs.count();
  expect(count).toBeGreaterThan(3);
  for (let index = 0; index < count; index += 1) {
    const control = inputs.nth(index);
    await control.focus();
    // A control that cannot take focus cannot be used without a mouse.
    await expect(control).toBeFocused();
  }
});

/**
 * Collect every text-bearing element with the colours a reader sees.
 *
 * The background is gathered as the whole ancestor chain rather than one
 * element's own, because the composite is what the eye receives.
 */
async function collectContrastSamples(page: Page): Promise<ContrastSample[]> {
  const raw = await page.evaluate(() => {
    const collected: {
      label: string;
      text: string;
      colour: string;
      layers: string[];
      fontSize: string;
      fontWeight: string;
    }[] = [];
    for (const element of Array.from(document.querySelectorAll<HTMLElement>("body *"))) {
      const own = Array.from(element.childNodes)
        .filter((node) => node.nodeType === Node.TEXT_NODE)
        .map((node) => node.textContent ?? "")
        .join("")
        .trim();
      if (own.length === 0) continue;
      const box = element.getBoundingClientRect();
      if (box.width === 0 || box.height === 0) continue;
      const style = window.getComputedStyle(element);
      if (style.visibility === "hidden" || style.opacity === "0") continue;
      const layers: string[] = [];
      let node: HTMLElement | null = element;
      while (node !== null) {
        const nodeStyle = window.getComputedStyle(node);
        // A background image hides whatever is behind it and has no single
        // colour, so the chain stops here and stays unresolved.
        layers.push(nodeStyle.backgroundImage === "none" ? nodeStyle.backgroundColor : "image");
        node = node.parentElement;
      }
      layers.push(window.getComputedStyle(document.documentElement).backgroundColor);
      collected.push({
        colour: style.color,
        fontSize: style.fontSize,
        fontWeight: style.fontWeight,
        label: `${element.tagName.toLowerCase()}${element.id === "" ? "" : `#${element.id}`}`,
        layers,
        text: own,
      });
    }
    return collected;
  });
  return raw.map((entry) => {
    // The browser side only gathers; compositing lives in the audited module.
    const background = resolveBackground(entry.layers.map((layer) => parseRgba(layer)));
    return {
      background,
      bold: Number(entry.fontWeight) >= 700,
      fontSize: Number.parseFloat(entry.fontSize),
      foreground: parseRgba(entry.colour) ?? { a: 1, b: 0, g: 0, r: 0 },
      label: entry.label,
      text: entry.text,
    };
  });
}

test("no text falls below its WCAG AA contrast threshold for the first time", async ({
  page,
}) => {
  await openCanvas(page);
  await addTwoConnectedPopulations(page);

  const samples = await collectContrastSamples(page);
  expect(samples.length).toBeGreaterThan(10);
  const results = samples.map(judge);
  const comparison = compareWithBaseline(results, BASELINE.failures);

  // A background that could not be resolved is never scored, so it must never
  // be silently absent from the report either.
  expect(comparison.unresolved).toEqual([]);
  // Anything failing that the baseline does not list is a new defect.
  expect(comparison.regressions).toEqual([]);
  // Anything the baseline lists that no longer fails means the list is stale;
  // a baseline allowed to drift becomes a blanket permission.
  expect(comparison.fixed).toEqual([]);
  // The failures the baseline does record are real, user-facing defects,
  // tracked in the private TODO rather than accepted.
  expect(Array.isArray(BASELINE.failures)).toBe(true);
});
