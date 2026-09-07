// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { federation } from "@module-federation/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const studioApiOrigin = process.env.SC_NEUROCORE_STUDIO_API_ORIGIN
  ?? "http://127.0.0.1:8001";

const studioApiProxy = {
  "/api": {
    target: studioApiOrigin,
    changeOrigin: true,
  },
} as const;

const reactSharedContract = {
  react: { singleton: true, requiredVersion: "19.2.7" },
  "react-dom": { singleton: true, requiredVersion: "19.2.7" },
} as const;

export default defineConfig({
  base: "/studios/sc-neurocore/",
  plugins: [
    react(),
    federation({
      name: "sc_neurocore",
      filename: "remoteEntry.js",
      dev: {
        disableDynamicRemoteTypeHints: true,
        disableHotTypesReload: true,
      },
      dts: {
        consumeTypes: false,
        generateTypes: true,
      },
      exposes: {
        "./SnnStudioPanel": "./src/SnnStudioPanel.tsx",
      },
      shared: reactSharedContract,
    }),
  ],
  build: {
    target: "esnext",
  },
  optimizeDeps: {
    // The dev server must re-optimise on every start rather than reuse the
    // cache in `node_modules/.vite`.
    //
    // Module federation serves React through its own shared-module graph. With
    // a cache warm from an earlier dev run, the two disagree about which copy
    // of `react/jsx-dev-runtime` is in force and the browser fails with
    // `_jsxDEV is not a function`. The symptom is unusually easy to
    // misattribute: a cold run passes, so the first thing anyone tries works,
    // and only the *second* run of the same suite fails. CI checks out fresh
    // and is therefore always cold, which is why this never showed there.
    //
    // Measured 2026-09-07 on the default Playwright suite: cold 5 of 5 passing,
    // warm 5 of 5 failing, and with this option cold, warm and warm-again all
    // 5 of 5. Excluding the shared packages from the optimiser was tried first
    // and did not fix it. `optimizeDeps` is dev-only; the production build is
    // untouched.
    force: true,
  },
  server: {
    proxy: studioApiProxy,
  },
  preview: {
    cors: true,
    // The built bundle calls a same-origin /api, so a preview that cannot
    // reach the Studio backend can only ever render an empty catalogue.
    proxy: studioApiProxy,
  },
});
