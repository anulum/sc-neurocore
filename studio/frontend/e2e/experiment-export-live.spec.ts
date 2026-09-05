// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — live browser-to-server-to-replay export contract

import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, test, type Page } from "@playwright/test";

const MODEL_NAME = "HodgkinHuxleyNeuron";
// The model's own default timestep is 0.01 ms and its default drive is
// constant. Asking the UI for neither is the case a syntax-only export got
// silently wrong.
const REQUESTED_DT = "0.05";
const REQUESTED_PROTOCOL = "step";

test.describe.configure({ mode: "serial" });

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
  });
});

async function openStudioWithTheModel(page: Page): Promise<void> {
  await page.goto("./");
  await page.getByPlaceholder("Search models...").fill(MODEL_NAME);
  const contract = page.getByTestId(`model-contract-${MODEL_NAME}`);
  await expect(contract).toBeVisible();
  const detail = page.waitForResponse(
    (response) =>
      new URL(response.url()).pathname === `/api/models/${MODEL_NAME}` && response.ok(),
  );
  await contract.locator("..").click();
  await detail;
}

async function chooseNondefaultExperiment(page: Page): Promise<void> {
  await page.getByTestId("protocol-select").selectOption(REQUESTED_PROTOCOL);
  await page.getByTestId("slider-dt").fill(REQUESTED_DT);
  await expect(page.getByTestId("protocol-select")).toHaveValue(REQUESTED_PROTOCOL);
}

test("the browser exports a script that states the experiment the server resolved", async ({
  page,
}) => {
  await openStudioWithTheModel(page);
  await chooseNondefaultExperiment(page);

  const codegen = page.waitForResponse(
    (response) => new URL(response.url()).pathname === "/api/codegen" && response.ok(),
  );
  await page.getByTestId("run-codegen").click();
  const payload = await (await codegen).json();

  expect(payload.request).toMatchObject({
    name: MODEL_NAME,
    dt: Number(REQUESTED_DT),
    protocol: REQUESTED_PROTOCOL,
    trial: "replay",
  });
  const script = await page.getByTestId("codegen-script").innerText();
  expect(script).toContain(payload.experiment_sha256);
  expect(script).toContain(`"dt": ${REQUESTED_DT}`);
  expect(script).toContain(`"protocol": "${REQUESTED_PROTOCOL}"`);
  // The assumptions the previous export hard-coded must not be back.
  expect(script).not.toContain("step(current=");
  expect(script).not.toContain("neuron.v");
  await expect(page.getByTestId("codegen-experiment")).toContainText(
    payload.experiment_sha256.slice(0, 16),
  );
});

test("a pack downloaded from the browser replays in a clean interpreter", async ({ page }) => {
  await openStudioWithTheModel(page);
  await chooseNondefaultExperiment(page);

  const download = page.waitForEvent("download");
  const packResponse = page.waitForResponse(
    (response) =>
      new URL(response.url()).pathname === "/api/export/replay-pack" && response.ok(),
  );
  await page.getByTestId("export-replay-pack").click();
  const served = await (await packResponse).json();
  const artefact = await download;

  expect(artefact.suggestedFilename()).toContain(
    served.experiment_identity_sha256.slice(0, 12),
  );

  const workspace = mkdtempSync(join(tmpdir(), "sc-neurocore-replay-"));
  const packPath = join(workspace, "pack.json");
  await artefact.saveAs(packPath);

  // What the browser wrote to disk is what the server sealed.
  const saved = JSON.parse(readFileSync(packPath, "utf-8"));
  expect(saved).toEqual(served);
  expect(saved.schema_version).toBe("studio.replay-pack.v1");
  expect(saved.request).toMatchObject({
    name: MODEL_NAME,
    dt: Number(REQUESTED_DT),
    protocol: REQUESTED_PROTOCOL,
  });

  // A separate interpreter, outside the repository, with no PYTHONPATH: the
  // pack has to carry everything the replay needs.
  const environment = { ...process.env };
  delete environment.PYTHONPATH;
  const replayed = execFileSync(
    "python",
    ["-m", "sc_neurocore.studio.replay_pack", packPath, "--json"],
    { cwd: workspace, env: environment, encoding: "utf-8", timeout: 900_000 },
  );
  const outcome = JSON.parse(replayed);

  expect(outcome.verdict).toBe("match");
  expect(outcome.differences).toEqual([]);
  expect(outcome.experiment_identity_sha256).toBe(served.experiment_identity_sha256);

  // A pack whose expectation no longer describes the run is reported, not
  // rounded away.
  const tampered = { ...saved };
  tampered.expectation = { ...saved.expectation, spike_count: saved.expectation.spike_count + 3 };
  const tamperedPath = join(workspace, "tampered.json");
  writeFileSync(tamperedPath, JSON.stringify(tampered), "utf-8");
  let status = 0;
  let output = "";
  try {
    output = execFileSync(
      "python",
      ["-m", "sc_neurocore.studio.replay_pack", tamperedPath],
      { cwd: workspace, env: environment, encoding: "utf-8", timeout: 900_000 },
    );
  } catch (failure) {
    const error = failure as { status: number; stdout: string };
    status = error.status;
    output = error.stdout;
  }
  expect(status).toBe(1);
  expect(output).toContain("verdict: mismatch");
});
