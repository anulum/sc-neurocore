// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — built Module Federation browser contract

import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  testMatch: "module-federation-host.spec.ts",
  timeout: 60_000,
  expect: {
    timeout: 15_000,
  },
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: "http://127.0.0.1:5185",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: [
    {
      command: "npm run preview -- --host 127.0.0.1 --port 5184 --strictPort",
      reuseExistingServer: false,
      timeout: 120_000,
      url: "http://127.0.0.1:5184/studios/sc-neurocore/remoteEntry.js",
    },
    {
      command: "vite --config e2e/federation-host/vite.config.ts",
      reuseExistingServer: false,
      timeout: 120_000,
      url: "http://127.0.0.1:5185",
    },
  ],
});
