// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — built-bundle browser contract for the Network Canvas

import { defineConfig, devices } from "@playwright/test";

/**
 * Return a port from the environment, refusing anything that is not one.
 *
 * A shared workstation runs several of these suites at once, so the ports are
 * configurable; a silent fallback on a malformed value would start the servers
 * somewhere the tests do not look.
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

const apiPort = configuredPort("SC_NEUROCORE_STUDIO_GRAPH_API_PORT", 18_004);
const uiPort = configuredPort("SC_NEUROCORE_STUDIO_GRAPH_UI_PORT", 15_178);
const apiOrigin = `http://127.0.0.1:${apiPort}`;
const uiOrigin = `http://127.0.0.1:${uiPort}`;

// The canvas contract runs against the BUILT bundle served by `vite preview`,
// which proxies /api to a real Studio backend — no mocked routes. What the
// canvas offers a keyboard and a screen reader is only worth checking against
// the graph the server actually validates and runs. Run `npm run build` first;
// the `test:e2e:graph` script does.
export default defineConfig({
  testDir: "./e2e",
  // Live-backend browser contracts over the built bundle: the network canvas
  // and the guided workflow's truthfulness under races and failures.
  testMatch: ["network-canvas-live.spec.ts", "guided-flow-truth-live.spec.ts"],
  // A shared workstation can be heavily loaded; the budget is for the
  // boundary, not for the host's spare capacity.
  timeout: 900_000,
  expect: { timeout: 60_000 },
  fullyParallel: false,
  workers: 1,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: `${uiOrigin}/studios/sc-neurocore/`,
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium-graph",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: [
    {
      command: `python -m uvicorn sc_neurocore.studio.app:create_app --factory --host 127.0.0.1 --port ${apiPort} --log-level warning`,
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
