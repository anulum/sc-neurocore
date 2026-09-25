// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — live network tutorial notebook, against a real backend

/**
 * The canvas's Notebook button downloads a tutorial that rebuilds the drawn
 * network with the public API; its seal is the Studio's run of that graph.
 * The Python suite runs such a notebook in a fresh process.
 */

import { readFileSync } from "node:fs";

import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
  });
});

test("the Notebook button downloads a tutorial sealed to the canvas run", async ({ page }) => {
  await page.goto("./");
  const models = page.waitForResponse(
    (response) => new URL(response.url()).pathname === "/api/graph/models" && response.ok(),
  );
  await page.getByRole("button", { name: "Canvas", exact: true }).first().click();
  await models;
  await page.getByRole("button", { name: "+ Exc", exact: true }).click();
  await page.getByRole("button", { name: "+ Inh", exact: true }).click();

  const simulated = page.waitForResponse(
    (response) => new URL(response.url()).pathname === "/api/graph/simulate" && response.ok(),
  );
  await page.getByRole("button", { name: "Simulate", exact: true }).click();
  const run = (await (await simulated).json()) as { n_spikes: number; spec: { graph_sha256: string } };

  const download = page.waitForEvent("download");
  await page.getByTestId("export-network-notebook").click();
  const file = await download;
  expect(file.suggestedFilename()).toBe("network-tutorial.ipynb");
  const notebook = JSON.parse(readFileSync(await file.path(), "utf8")) as {
    nbformat: number;
    metadata: { sc_neurocore: { kind: string; graph_sha256: string; n_spikes: number } };
    cells: { cell_type: string; source: string }[];
  };
  expect(notebook.nbformat).toBe(4);
  const sealed = notebook.metadata.sc_neurocore;
  expect(sealed.kind).toBe("sc-neurocore.network-tutorial.v1");
  expect(sealed.graph_sha256).toBe(run.spec.graph_sha256);
  expect(sealed.n_spikes).toBe(run.n_spikes);
  const code = notebook.cells.filter((cell) => cell.cell_type === "code").map((cell) => cell.source).join("\n");
  expect(code).toContain("= Population(");
  expect(code).toContain('network.run(duration=N_STEPS * DT_S, dt=DT_S, backend="python")');
  expect(code).toContain('print("spikes match:"');
  expect(code).not.toMatch(/\/home\/|\/tmp\//);
  await expect(page.getByText("Notebook written: it rebuilds this network step by step")).toBeVisible();
});
