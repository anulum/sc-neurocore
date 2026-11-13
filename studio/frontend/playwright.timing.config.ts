// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — render timing measurement of the built Studio against the real backend

import { defineConfig, devices } from "@playwright/test";

/**
 * Return a port from the environment, refusing anything that is not one.
 *
 * @param name - Environment variable that may hold the port.
 * @param fallback - Port to use when it is unset.
 * @returns The port to bind.
 * @throws {Error} When the variable holds something that is not a port.
 */
function configuredPort(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined) return fallback;
  const port = Number(raw);
  if (!Number.isInteger(port) || port < 1 || port > 65_535) {
    throw new Error(`${name} must be an integer port in the range 1..65535.`);
  }
  return port;
}

const apiPort = configuredPort("SC_NEUROCORE_STUDIO_TIMING_API_PORT", 18_006);
const uiPort = configuredPort("SC_NEUROCORE_STUDIO_TIMING_UI_PORT", 15_180);
const apiOrigin = `http://127.0.0.1:${apiPort}`;
const uiOrigin = `http://127.0.0.1:${uiPort}`;

// What is measured is the BUILT bundle served by `vite preview` against a real
// Studio backend: a development server would time React's development build,
// which a user never runs. Run `npm run build` first; `test:e2e:timing` does.
// The run is long (ten samples of every workload) and is never part of the
// regular browser suites.
export default defineConfig({
  testDir: "./e2e",
  testMatch: ["render-timing-live.spec.ts"],
  timeout: 1_800_000,
  expect: { timeout: 120_000 },
  fullyParallel: false,
  workers: 1,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: `${uiOrigin}/studios/sc-neurocore/`,
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium-timing",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: [
    {
      command: `python -m uvicorn sc_neurocore.studio.app:create_app --factory --host 127.0.0.1 --port ${apiPort} --log-level warning`,
      env: {
        SC_NEUROCORE_STUDIO_CORS_ORIGINS: [apiOrigin, uiOrigin].join(","),
        SC_NEUROCORE_STUDIO_WEBSOCKET_ALLOWED_ORIGINS: [apiOrigin, uiOrigin].join(","),
      },
      reuseExistingServer: false,
      timeout: 120_000,
      url: `${apiOrigin}/api/health`,
    },
    {
      command: `npm run preview -- --host 127.0.0.1 --port ${uiPort} --strictPort`,
      env: {
        SC_NEUROCORE_STUDIO_API_ORIGIN: apiOrigin,
      },
      reuseExistingServer: false,
      timeout: 120_000,
      url: `${uiOrigin}/studios/sc-neurocore/`,
    },
  ],
});
