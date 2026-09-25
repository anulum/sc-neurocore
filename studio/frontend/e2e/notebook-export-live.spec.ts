// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — live notebook export, against a real backend

/**
 * The Notebook button downloads a cited notebook that carries the sealed pack
 * inline; the Python suite runs such a notebook in a fresh process.
 */

import { readFileSync } from "node:fs";

import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
  });
});

test("the Notebook button downloads a cited notebook carrying the sealed pack", async ({ page }) => {
  const simulated = page.waitForResponse(
    (response) => new URL(response.url()).pathname === "/api/models/simulate" && response.ok(),
  );
  await page.goto("./");
  await simulated;

  const download = page.waitForEvent("download");
  await page.getByTestId("export-replay-notebook").click();
  const file = await download;
  expect(file.suggestedFilename()).toMatch(/-replay\.ipynb$/);
  const notebook = JSON.parse(readFileSync(await file.path(), "utf8")) as {
    nbformat: number;
    cells: { cell_type: string; source: string }[];
  };
  expect(notebook.nbformat).toBe(4);
  const [intro, load, replay] = notebook.cells;
  expect(intro?.source).toContain("**Source.**");
  expect(intro?.source).toContain("establishes nothing about hardware");
  expect(load?.source).toContain("PACK = json.loads(");
  expect(load?.source).not.toMatch(/\/home\/|\/tmp\//);
  expect(replay?.source).toContain("replay_pack(PACK, allow_runtime_drift=True)");
});
