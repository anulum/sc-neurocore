// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  // Live-backend specs run under their own configs (graph, live, event, export),
  // which start the real API they need; the mocked default run cannot serve them.
  testIgnore: [
    "candidate-authoring-live.spec.ts",
    "catalogue-to-silicon-live.spec.ts",
    "event-training-live.spec.ts",
    "experiment-export-live.spec.ts",
    "fit-live.spec.ts",
    "guided-flow-truth-live.spec.ts",
    "module-federation-host.spec.ts",
    "network-canvas-live.spec.ts",
    "network-notebook-live.spec.ts",
    "notebook-export-live.spec.ts",
    "review-live.spec.ts",
    "training-conversion-live.spec.ts",
    "training-preregistration-live.spec.ts",
    "workbench-accessibility-live.spec.ts",
  ],
  timeout: 30_000,
  expect: {
    timeout: 15_000,
  },
  fullyParallel: true,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: "http://127.0.0.1:5174",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 5174",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    url: "http://127.0.0.1:5174",
  },
});
