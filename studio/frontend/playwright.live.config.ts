// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — live FastAPI-to-browser catalogue-to-silicon contract

import { tmpdir } from "node:os";
import { join } from "node:path";

import { defineConfig, devices } from "@playwright/test";

function configuredPort(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined) return fallback;
  const port = Number(raw);
  if (!Number.isInteger(port) || port < 1 || port > 65_535) {
    throw new Error(`${name} must be an integer port in the range 1..65535.`);
  }
  return port;
}

const apiPort = configuredPort("SC_NEUROCORE_STUDIO_LIVE_API_PORT", 18_001);
const uiPort = configuredPort("SC_NEUROCORE_STUDIO_LIVE_UI_PORT", 15_174);
const apiOrigin = `http://127.0.0.1:${apiPort}`;
const uiOrigin = `http://127.0.0.1:${uiPort}`;
const jobRoot = process.env.SC_NEUROCORE_STUDIO_LIVE_JOB_ROOT
  ?? join(tmpdir(), `sc-neurocore-studio-live-e2e-${process.pid}`);
const auditLogPath = process.env.SC_NEUROCORE_STUDIO_LIVE_AUDIT_LOG_PATH
  ?? join(tmpdir(), `sc-neurocore-studio-live-e2e-audit-${process.pid}.jsonl`);

export default defineConfig({
  testDir: "./e2e",
  testMatch: "catalogue-to-silicon-live.spec.ts",
  timeout: 240_000,
  expect: {
    timeout: 30_000,
  },
  fullyParallel: false,
  workers: 1,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: `${uiOrigin}/studios/sc-neurocore/`,
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium-live",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: [
    {
      command: `python -m uvicorn sc_neurocore.studio.app:create_app --factory --host 127.0.0.1 --port ${apiPort} --log-level warning`,
      env: {
        SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH: auditLogPath,
        SC_NEUROCORE_STUDIO_JOB_ROOT: jobRoot,
      },
      reuseExistingServer: false,
      timeout: 120_000,
      url: `${apiOrigin}/api/health`,
    },
    {
      command: `npm run dev -- --host 127.0.0.1 --port ${uiPort} --strictPort`,
      env: {
        SC_NEUROCORE_STUDIO_API_ORIGIN: apiOrigin,
      },
      reuseExistingServer: false,
      timeout: 120_000,
      url: `${uiOrigin}/studios/sc-neurocore/`,
    },
  ],
});
