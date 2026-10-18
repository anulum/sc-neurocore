// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Installed-wheel browser and independent replay acceptance

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { delimiter, dirname, isAbsolute, join } from "node:path";

import { defineConfig, devices } from "@playwright/test";

const python = process.env.SC_NEUROCORE_STUDIO_INSTALLED_PYTHON;
if (python === undefined || !isAbsolute(python) || !existsSync(python)) {
  throw new Error("SC_NEUROCORE_STUDIO_INSTALLED_PYTHON must name an existing absolute installed Python.");
}
const root = process.env.SC_NEUROCORE_STUDIO_INSTALLED_TEST_ROOT;
if (root === undefined || !isAbsolute(root)) {
  throw new Error("SC_NEUROCORE_STUDIO_INSTALLED_TEST_ROOT must name an absolute test directory.");
}
const launcher = join(dirname(python), "sc-neurocore");
if (!existsSync(launcher)) throw new Error("The installed environment has no sc-neurocore launcher.");
const port = Number(process.env.SC_NEUROCORE_STUDIO_INSTALLED_PORT ?? "18005");
if (!Number.isInteger(port) || port < 1 || port > 65_535) {
  throw new Error("SC_NEUROCORE_STUDIO_INSTALLED_PORT must be an integer port in 1..65535.");
}
mkdirSync(root, { recursive: true });
// Refuse editable source imports before browser success can be mistaken for wheel acceptance.
execFileSync(python, ["-I", "-c", [
  "from pathlib import Path",
  "import sys, sc_neurocore",
  "assert Path(sc_neurocore.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), 'Studio imports outside the installed environment'",
].join("\n")], { cwd: root, timeout: 30_000 });

const origin = `http://127.0.0.1:${port}`;
const path = `${dirname(python)}${delimiter}${process.env.PATH ?? ""}`;

export default defineConfig({
  testDir: "./e2e",
  testMatch: ["catalogue-to-silicon-live.spec.ts", "experiment-export-live.spec.ts"],
  grep: /Poisson constructor controls run, replay, export and preserve observed state|a full-domain seed survives serialized project custody, editing, local sessions and independent replay/,
  timeout: 900_000,
  expect: { timeout: 60_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  outputDir: join(root, "playwright"),
  reporter: process.env.CI ? "github" : "list",
  use: { baseURL: `${origin}/studios/sc-neurocore/`, trace: "retain-on-failure" },
  projects: [{
    name: "installed-chromium",
    use: {
      ...devices["Desktop Chrome"],
      launchOptions: { executablePath: process.env.SC_NEUROCORE_STUDIO_BROWSER_EXECUTABLE },
    },
  }],
  webServer: {
    command: `sc-neurocore studio --port ${port}`,
    env: {
      PATH: path,
      PYTHONPATH: "",
      PYTHONHOME: "",
      BROWSER: "true",
      SC_NEUROCORE_STUDIO_JOB_ROOT: join(root, "jobs"),
      SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH: join(root, "audit.jsonl"),
      SC_NEUROCORE_STUDIO_WEBSOCKET_ALLOWED_ORIGINS: origin,
    },
    reuseExistingServer: false,
    timeout: 120_000,
    url: `${origin}/api/health`,
  },
});
