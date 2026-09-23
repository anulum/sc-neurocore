// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — live browser-to-server-to-replay export contract

import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";

import { expect, test, type Page } from "@playwright/test";

import type { CodegenResponse, ReplayPack } from "../src/api/types";

/**
 * What `python -m sc_neurocore.studio.replay_pack --json` prints.
 *
 * The runner's contract, not the browser's, so it is stated here rather than
 * in `src/api/types.ts`: nothing the Studio ships reads this document.
 */
interface ReplayOutcome {
  /** `"match"` when the replay reproduced the sealed expectation. */
  verdict: string;
  /** Every field that did not reproduce, empty on a match. */
  differences: readonly unknown[];
  /** The identity the pack was sealed under, echoed back. */
  experiment_identity_sha256: string;
}

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

/** Open the Studio on the model this receipt is about, with its detail loaded. */
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
  await expect(page.getByTestId("slider-g_na")).toBeVisible();
  await expect(page.getByTestId("slider-g_a")).not.toBeVisible();
}

/**
 * Ask for a timestep and protocol that are not the model's own defaults.
 *
 * A syntax-only export agreed with the defaults by accident; asking for
 * neither is the case that caught it.
 */
async function chooseNondefaultExperiment(page: Page): Promise<void> {
  await page.getByTestId("protocol-select").selectOption(REQUESTED_PROTOCOL);
  await page.getByTestId("slider-dt").fill(REQUESTED_DT);
  await expect(page.getByTestId("protocol-select")).toHaveValue(REQUESTED_PROTOCOL);
}

test("the browser exports a script that states the experiment the server resolved", async ({
  page,
}, testInfo) => {
  await openStudioWithTheModel(page);
  await chooseNondefaultExperiment(page);

  const codegen = page.waitForResponse(
    (response) => new URL(response.url()).pathname === "/api/codegen" && response.ok(),
  );
  await page.getByTestId("run-codegen").click();
  const payload = (await (await codegen).json()) as CodegenResponse;

  expect(payload.request).toMatchObject({
    name: MODEL_NAME,
    dt: Number(REQUESTED_DT),
    protocol: REQUESTED_PROTOCOL,
    trial: "replay",
  });
  const script = await page.getByTestId("codegen-script").innerText();
  expect(script).toContain(payload.experiment_sha256);
  const scriptPath = testInfo.outputPath("experiment.py");
  writeFileSync(scriptPath, script, "utf-8");
  const environment = { ...process.env };
  delete environment.PYTHONPATH;
  // Execute exactly what the code panel exposes, including digest admission.
  // The interpreter may use an editable install; wheel isolation is verified
  // separately by test_studio_distribution.py.
  const output = execFileSync("python", ["-c", `
import contextlib, io, json, runpy, sys
from sc_neurocore.studio.replay_pack import replay_expectation
with contextlib.redirect_stdout(io.StringIO()):
    exported = runpy.run_path(sys.argv[1], run_name="__main__")
print(json.dumps({"request": exported["REQUEST"],
                  "expectation": replay_expectation(exported["result"])}))
`, scriptPath], {
    cwd: testInfo.outputDir, env: environment, encoding: "utf-8", timeout: 120_000,
  });
  const executed: unknown = JSON.parse(output);
  const reference = page.waitForResponse(
    (response) => new URL(response.url()).pathname === "/api/export/replay-pack" && response.ok(),
  );
  await page.getByTestId("export-replay-pack").click();
  const pack = (await (await reference).json()) as ReplayPack;
  expect(executed).toEqual({ request: payload.request, expectation: pack.expectation });
  // The assumptions the previous export hard-coded must not be back.
  expect(script).not.toContain("step(current=");
  expect(script).not.toContain("neuron.v");
  await expect(page.getByTestId("codegen-experiment")).toContainText(
    payload.experiment_sha256.slice(0, 16),
  );
});

test("a downloaded pack replays in a separate interpreter and reports tampering", async ({ page }, testInfo) => {
  await openStudioWithTheModel(page);
  await chooseNondefaultExperiment(page);

  const download = page.waitForEvent("download");
  const packResponse = page.waitForResponse(
    (response) =>
      new URL(response.url()).pathname === "/api/export/replay-pack",
    { timeout: 60_000 },
  );
  await page.getByTestId("export-replay-pack").click();
  const response = await packResponse;
  const failureBody = response.ok() ? "" : await response.text();
  expect(response.status(), failureBody).toBe(200);
  const served = (await response.json()) as ReplayPack;
  const artefact = await download;

  expect(artefact.suggestedFilename()).toContain(
    served.experiment_identity_sha256.slice(0, 12),
  );

  const workspace = testInfo.outputDir;
  const packPath = testInfo.outputPath("pack.json");
  await artefact.saveAs(packPath);

  // What the browser wrote to disk is what the server sealed.
  const saved = JSON.parse(readFileSync(packPath, "utf-8")) as ReplayPack;
  expect(saved).toEqual(served);
  expect(saved.schema_version).toBe("studio.replay-pack.v2");
  expect(saved.request).toMatchObject({
    name: MODEL_NAME,
    dt: Number(REQUESTED_DT),
    protocol: REQUESTED_PROTOCOL,
  });

  // A separate interpreter, outside the repository, with no PYTHONPATH: the
  // pack carries the replay inputs. Editable hooks can remain; this is not
  // the separate installed-wheel receipt in test_studio_distribution.py.
  const environment = { ...process.env };
  delete environment.PYTHONPATH;
  const replayed = execFileSync(
    "python",
    ["-m", "sc_neurocore.studio.replay_pack", packPath, "--json"],
    { cwd: workspace, env: environment, encoding: "utf-8", timeout: 900_000 },
  );
  const outcome = JSON.parse(replayed) as ReplayOutcome;

  expect(outcome.verdict).toBe("match");
  expect(outcome.differences).toEqual([]);
  expect(outcome.experiment_identity_sha256).toBe(served.experiment_identity_sha256);

  // A pack whose expectation no longer describes the run is reported, not
  // rounded away.
  // `expectation` travels opaquely to the Python runner, so its fields are
  // `unknown` here. Reading the count through a check rather than a cast keeps
  // the tamper honest: if the pack stops carrying a numeric spike count, this
  // fails loudly instead of writing `NaN` into the file and asserting on it.
  const spikeCount = saved.expectation.spike_count;
  if (typeof spikeCount !== "number") {
    throw new TypeError(`the pack's expectation carries no numeric spike_count: ${typeof spikeCount}`);
  }
  const tampered: ReplayPack = {
    ...saved,
    expectation: { ...saved.expectation, spike_count: spikeCount + 3 },
  };
  const tamperedPath = testInfo.outputPath("tampered.json");
  writeFileSync(tamperedPath, JSON.stringify(tampered), "utf-8");
  let status = 0;
  let output: string;
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
