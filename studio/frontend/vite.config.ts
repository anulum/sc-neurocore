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
  server: {
    proxy: {
      "/api": {
        target: studioApiOrigin,
        changeOrigin: true,
      },
    },
  },
  preview: {
    cors: true,
  },
});
